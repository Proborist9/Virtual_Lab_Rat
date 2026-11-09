"""
Virtual Rat Sandbox - minimal working prototype (v0.1)
Scenario: kidney transplant -> immune rejection -> cytokine storm -> BP drop -> organ failure.

ALL PARAMETERS ARE PLACEHOLDERS for demonstration. Before any scientific use they
must be replaced with literature values and calibrated against real rat data.

Run:  python virtual_rat_sandbox.py
Needs: numpy, scipy (matplotlib optional, for the graph)
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------- 1. PARAMETERS
@dataclass
class Params:
    # immune module
    t_prolif: float = 1.2       # effector T-cell growth rate (1/day)
    t_death: float = 0.35       # T-cell death rate (1/day)
    t_max: float = 10.0         # carrying capacity (arbitrary units)
    mismatch: float = 0.8       # donor/recipient mismatch (0 = identical, 1 = maximum)
    drug_ic50: float = 0.5      # drug level that halves T-cell growth
    drug_decay: float = 0.9     # drug clearance (1/day)
    infusion: float = 0.0       # drug input rate (set by the experiment runner)
    # graft (kidney) module
    kill_rate: float = 0.12     # graft killing by T cells
    repair_rate: float = 0.05   # regeneration (1/day)
    # cytokine module
    cyt_from_t: float = 0.8
    cyt_from_damage: float = 1.5
    cyt_clear: float = 1.0
    # kidney function (creatinine)
    cr_clear: float = 4.0       # 1/day
    cr_base: float = 0.5        # mg/dL healthy rat (placeholder)
    # hemodynamics
    map_base: float = 100.0     # normal mean arterial pressure (mmHg)
    map_drop_max: float = 55.0  # max cytokine-driven drop (mmHg)
    cyt_half: float = 4.0       # cytokine level giving half the max drop
    hill_n: float = 2.0
    map_tau: float = 0.3        # BP relaxation time (days)
    # liver
    liver_injury: float = 1.5
    liver_recov: float = 0.3

# ---------------------------------------------------------------- 2. STATE
NAMES = ["Teff", "Cyt", "Graft", "Creat", "MAP", "Liver", "Drug"]
IDX = {n: i for i, n in enumerate(NAMES)}
LOWER = np.array([0, 0, 0, 0.1, 0, 0, 0])        # physical floors
UPPER = np.array([50, 100, 1, 30, 200, 1, 50])   # physical ceilings

def hill(x, k, n):
    x = max(x, 0.0)
    return x**n / (k**n + x**n)

# ---------------------------------------------------------------- 3. MODEL
def rhs(t, y, p: Params):
    y = np.clip(y, LOWER, UPPER)                 # never leave physical range
    Teff, Cyt, G, Cr, MAP, Liv, Drug = y

    suppress = 1.0 / (1.0 + Drug / p.drug_ic50)  # immunosuppression
    dTeff = (p.t_prolif * p.mismatch * G * suppress * (1 - Teff / p.t_max)
             - p.t_death * Teff)
    dG = -p.kill_rate * Teff * G + p.repair_rate * (1 - G) * G
    dCyt = p.cyt_from_t * Teff + p.cyt_from_damage * (1 - G) - p.cyt_clear * Cyt
    dCr = p.cr_base * p.cr_clear - p.cr_clear * G * Cr
    map_target = p.map_base - p.map_drop_max * hill(Cyt, p.cyt_half, p.hill_n)
    dMAP = (map_target - MAP) / p.map_tau
    perfusion = min(1.0, hill(MAP, 60.0, 4.0) / hill(p.map_base, 60.0, 4.0))
    dLiv = -p.liver_injury * (1 - perfusion) * Liv + p.liver_recov * (1 - Liv) * perfusion
    dDrug = p.infusion - p.drug_decay * Drug
    return [dTeff, dCyt, dG, dCr, dMAP, dLiv, dDrug]

# ---------------------------------------------------------------- 4. EVENTS
def make_events():
    def death(t, y, p): return y[IDX["MAP"]] - 40.0      # circulatory collapse
    def liver_fail(t, y, p): return y[IDX["Liver"]] - 0.15
    death.terminal = liver_fail.terminal = True
    return [death, liver_fail]

# ---------------------------------------------------------------- 5. EXPERIMENT RUNNER
@dataclass
class Experiment:
    days: float = 20.0
    drug_dose: float = 5.0
    drug_stop_day: float | None = 6.0    # None = continuous dosing
    params: Params = field(default_factory=Params)

def run(exp: Experiment):
    p = exp.params
    p.infusion = exp.drug_dose * p.drug_decay
    y0 = np.array([0.05, 0.0, 1.0, p.cr_base, p.map_base, 1.0, exp.drug_dose])
    t_all, y_all, status = [], [], "survived"
    segments = [(0.0, exp.drug_stop_day or exp.days)]
    if exp.drug_stop_day and exp.drug_stop_day < exp.days:
        segments.append((exp.drug_stop_day, exp.days))
    y_cur = y0
    for i, (a, b) in enumerate(segments):
        if i == 1:
            p.infusion = 0.0                     # scheduled event: stop the drug
        sol = solve_ivp(rhs, (a, b), y_cur, args=(p,), method="LSODA",
                        events=make_events(), rtol=1e-6, atol=1e-8, max_step=0.1)
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        t_all.append(sol.t); y_all.append(sol.y)
        y_cur = np.clip(sol.y[:, -1], LOWER, UPPER)
        if sol.status == 1:
            status = f"DIED at day {sol.t[-1]:.1f}"
            break
    return np.concatenate(t_all), np.concatenate(y_all, axis=1), status

# ---------------------------------------------------------------- 6. MAIN
if __name__ == "__main__":
    scenarios = {
        "No drug (untreated)": Experiment(drug_dose=0.0, drug_stop_day=None),
        "Drug stopped day 6":  Experiment(drug_dose=5.0, drug_stop_day=6.0),
        "Drug continuous":     Experiment(drug_dose=5.0, drug_stop_day=None),
    }
    results = {}
    for name, e in scenarios.items():
        t, y, status = run(e)
        results[name] = (t, y)
        print(f"\n{name}: {status}")
        for day in (2, 5, 10, 15):
            j = np.searchsorted(t, day)
            if j < len(t):
                row = " ".join(f"{n}={y[IDX[n], j]:.2f}" for n in ("Graft", "Creat", "MAP", "Liver"))
                print(f"  day {day:>2}: {row}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))
        for name, (t, y) in results.items():
            for a, n in zip(ax, ("Graft", "Creat", "MAP", "Liver")):
                a.plot(t, y[IDX[n]], label=name)
                a.set_title(n)
                a.set_xlabel("day")
        ax[0].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig("rejection_cascade.png", dpi=120)   # saved in the same folder as this file
        print("\nGraph saved as rejection_cascade.png")
    except ImportError:
        print("\nmatplotlib not installed, so no graph was drawn (results above are still valid).")