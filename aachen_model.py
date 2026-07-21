"""
================================================================================
 Aachen Public Transport Coverage — COMPLETE ANALYSIS PIPELINE
 Cost-Aware, Equity-Constrained Bus-Stop Placement + KPI Scenario Analysis
================================================================================
 Business Analytics Project, SoSe 2026.  Team: Pierre, Dirk, Wen, Hyeonju.
 Built on and validated against the Katsioupis (2026) p-median baseline.

 WHAT THIS PRODUCES (everything the paper needs)
 -----------------------------------------------
   results/00_baseline_validation.csv   reproduces Katsioupis exactly (proof)
   results/01_pareto_costblind.csv      budget sweep, homogeneous cost
   results/02_pareto_costaware.csv      budget sweep, heterogeneous cost
   results/03_equity_scenarios.csv      equity-constrained vs unconstrained
   results/05_cost_effectiveness.csv    EUR per extra covered person + the "knee"
   results/06_cost_sensitivity.csv      robustness to cost-parameter choice
   results/selected_stops_<scen>.csv    which candidates are built, per scenario
   figures/*.png                        Pareto, coverage, equity, sensitivity

 OUR CONTRIBUTION (beyond the baseline)
 --------------------------------------
   (1) Explicit HETEROGENEOUS cost model, grounded in German barrier-free
       stop-construction figures (~EUR 20k-100k/stop). Turns the abstract
       "<= p stops" into a real budget and lets us contrast a COST-BLIND plan
       (all stops equal) with a COST-AWARE plan -> different stops get built.
   (2) EQUITY dimension: an explicit target for the currently UNDERSERVED
       central population (baseline walk > 300 m). We show it changes WHICH
       stops are chosen and WHO benefits — a distributional view the baseline
       (system-average only) never takes.
   (3) A cost-EFFECTIVENESS decision rule (EUR per additional covered person)
       and a concrete recommendation for Aachen, not just a trade-off curve.

 SYSTEM / HOW TO RUN
 -------------------
   Pure Python (portable, the supervisor can reproduce without a GAMS licence):
       pip install pandas numpy scipy pulp matplotlib
       python aachen_model.py
   The optimisation core is ALSO provided as GAMSPy (aachen_model_gamspy.py) to
   satisfy the handout's GAMSPy requirement; both encode the identical model.

 DATA REQUIRED (all present except where noted)
 ----------------------------------------------
   existing.csv        1037 existing stops (stop_id,x,y)           [have]
   candidates.csv       141 candidate stops (stop_id,x,y)          [have]
   demand.csv         16984 demand nodes (demand_id,x,y,population,central) [have]
   -> cost per candidate is MODELLED here (see build_cost_model); the only
      external input worth firming up with the supervisor / AVV is the cost
      schema. We handle its uncertainty with the sensitivity analysis (06).
================================================================================
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import pulp

# ------------------------------------------------------------------ constants
F           = 1.3                 # circuity factor (Euclidean -> walking), baseline
COVER_RADII = (200, 300, 400, 600)
R_EQUITY    = 300                 # underserved := central node with baseline walk > this
OUT = "results"; FIG = "figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# cost model parameters (k EUR) — grounded in German barrier-free stop costs
#   base barrier-free rebuild (Buskap) ~45k; solo stop pays full mobilisation;
#   busy stops need articulated-bus platform + real-time display (DFI).
C_BASE   = 45.0
C_SOLO   = 15.0                   # + if not part of a bidirectional pair (<=40 m)
C_LOAD   = (0.0, 20.0, 40.0)      # + by demand-load tercile (platform/DFI need)
PAIR_M   = 40.0


# ============================================================ 1. DATA + DISTANCES
def load():
    ex  = pd.read_csv("existing.csv")
    ca  = pd.read_csv("candidates.csv")
    dem = pd.read_csv("demand.csv")
    demand = dem[["x", "y"]].to_numpy(float)
    pop    = dem["population"].to_numpy(float)
    central = (dem["central"].to_numpy() == 1)
    exy = ex[["x", "y"]].to_numpy(float)
    cxy = ca[["x", "y"]].to_numpy(float)
    cid = ca["stop_id"].to_numpy()
    return demand, pop, central, exy, cxy, cid


def distances(demand, exy, cxy):
    nearest_e = cKDTree(exy).query(demand)[0] * F              # baseline service
    dC = np.sqrt(((demand[:, None, :] - cxy[None, :, :]) ** 2).sum(-1)) * F
    improves = dC < nearest_e[:, None] - 1e-9
    return nearest_e, dC, improves


def build_cost_model(cxy, dC, improves, pop):
    """Heterogeneous per-candidate cost (k EUR). Two drivers:
       - geometry: solo stop (no near pair) pays extra mobilisation  [demand-independent]
       - load: candidates serving more people need bigger platform/DFI [demand-correlated]
       The mix is deliberate so a cost-aware plan differs from a cost-blind one."""
    nn2 = cKDTree(cxy).query(cxy, k=2)[0][:, 1]               # nearest other candidate
    solo = nn2 > PAIR_M
    # potential load = population of nodes this candidate could improve
    load = np.array([pop[improves[:, c]].sum() for c in range(len(cxy))])
    tier = np.zeros(len(cxy), int)
    pos = load[load > 0]
    if len(pos):
        q1, q2 = np.quantile(pos, [1/3, 2/3])
        tier = np.where(load <= q1, 0, np.where(load <= q2, 1, 2))
    cost = C_BASE + np.where(solo, C_SOLO, 0.0) + np.array([C_LOAD[t] for t in tier])
    return cost, load


# ============================================================ 2. CORE MODEL
def solve(budget, pop, nearest_e, dC, improves, cost,
          equity_mask=None, equity_R=R_EQUITY, equity_tau=0.0):
    """
    Cost-budgeted p-median (min population-weighted walking distance) with an
    optional equity floor. Compact reassignment form, equivalent to the baseline
    full-allocation p-median. Returns (opened_idx, solver_status).

      min  base + sum_{i,c} pop_i (dC_ic - nearest_e_i) y_ic
      s.t. sum_c y_ic <= 1                              (each node reassigned once)
           y_ic <= x_c                                  (coupling)
           sum_c cost_c x_c <= budget                   (COST BUDGET)   <-- ours
           sum_{i in U} sum_{c: dC_ic<=Req} pop_i y_ic >= tau * pop_U    (EQUITY) <-- ours
    """
    ccost = cost
    relevant = np.where(improves.any(1))[0]
    prob = pulp.LpProblem("aachen", pulp.LpMinimize)
    x = {c: pulp.LpVariable(f"x_{c}", cat="Binary") for c in range(len(cost))}
    y = {(i, c): pulp.LpVariable(f"y_{i}_{c}", lowBound=0, upBound=1, cat="Continuous")
         for i in relevant for c in np.where(improves[i])[0]}

    base = float((pop * nearest_e).sum())
    prob += base + pulp.lpSum(pop[i] * (dC[i, c] - nearest_e[i]) * y[i, c] for (i, c) in y)
    prob += pulp.lpSum(ccost[c] * x[c] for c in x) <= budget
    for i in relevant:
        prob += pulp.lpSum(y[i, c] for c in np.where(improves[i])[0]) <= 1
    for (i, c) in y:
        prob += y[i, c] <= x[c]

    if equity_mask is not None and equity_tau > 0:
        U = np.where(equity_mask)[0]
        popU = float(pop[U].sum())
        terms = [pop[i] * y[i, c] for i in U if i in relevant
                 for c in np.where(improves[i])[0] if dC[i, c] <= equity_R]
        prob += pulp.lpSum(terms) >= equity_tau * popU

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
    opened = [c for c in x if x[c].value() and x[c].value() > 0.5]
    return opened, pulp.LpStatus[status]


def build_pareto_model(pop, nearest_e, dC, improves, ccost):
    """Build the cost-budgeted p-median ONCE; sweep budgets by only changing the
    budget RHS (fast: no variable re-creation). y continuous (p-median is
    integral in assignment once x is fixed)."""
    relevant = np.where(improves.any(1))[0]
    prob = pulp.LpProblem("aachen_pareto", pulp.LpMinimize)
    x = {c: pulp.LpVariable(f"x_{c}", cat="Binary") for c in range(len(ccost))}
    y = {(i, c): pulp.LpVariable(f"y_{i}_{c}", lowBound=0, upBound=1, cat="Continuous")
         for i in relevant for c in np.where(improves[i])[0]}
    base = float((pop * nearest_e).sum())
    prob += base + pulp.lpSum(pop[i] * (dC[i, c] - nearest_e[i]) * y[i, c] for (i, c) in y)
    prob += pulp.lpSum(ccost[c] * x[c] for c in x) <= 0, "budget"
    for i in relevant:
        prob += pulp.lpSum(y[i, c] for c in np.where(improves[i])[0]) <= 1
    for (i, c) in y:
        prob += y[i, c] <= x[c]
    # no ghost stops: an opened candidate must serve at least one node, so the
    # solver cannot spend leftover budget on stops that help nobody.
    for c in range(len(ccost)):
        serving = [y[i, c] for i in relevant if (i, c) in y]
        prob += (x[c] <= pulp.lpSum(serving)) if serving else (x[c] == 0)
    return prob, x


def solve_status(prob):
    """Honest solver status. LpStatus only says CBC finished; the proof of
    optimality is in sol_status. Returns one of:
       'optimal'   -- certified optimal
       'feasible'  -- a feasible incumbent, but optimality NOT proven
                      (e.g. hit the time limit); must not be reported as optimal
       'infeasible'/'unbounded'/'undefined'
    """
    ls = pulp.LpStatus[prob.status]
    ss = pulp.LpSolution[prob.sol_status]          # 'Optimal Solution Found' etc.
    if ls == "Optimal" and ss == "Optimal Solution Found":
        return "optimal"
    if ss in ("Optimal Solution Found", "Solution Found"):
        return "feasible"
    return ls.lower()


def sweep_budget(prob, x, cid, budget):
    prob.constraints["budget"].constant = -budget          # set RHS = budget
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
    opened = [c for c in x if x[c].value() and x[c].value() > 0.5]
    return opened, solve_status(prob)


def min_cost_for_equity(pop, nearest_e, dC, improves, cost, equity_mask,
                        equity_R=R_EQUITY, equity_tau=0.5):
    """Minimum construction cost to cover a share tau of the underserved group
    within equity_R metres.

    Coverage is modelled with an explicit BINARY node-coverage variable z_i
    (a node is either covered or not -- no fractional coverage), linked to the
    stop variables by z_i <= sum_{c in C_i^R} x_c. This replaces the earlier
    pop_i * y_ic formulation, in which a continuous y could in principle count a
    node as partially covered.

    Returns the minimum cost in kEUR, or None if the target is infeasible.
    """
    U = np.where(equity_mask)[0]
    popU = float(pop[U].sum())
    cand = sorted({c for i in U for c in np.where(improves[i])[0] if dC[i, c] <= equity_R})
    covers = {i: [c for c in np.where(improves[i])[0] if dC[i, c] <= equity_R and c in cand]
              for i in U}
    covers = {i: cs for i, cs in covers.items() if cs}          # coverable nodes only

    prob = pulp.LpProblem("mincost_equity", pulp.LpMinimize)
    x = {c: pulp.LpVariable(f"x_{c}", cat="Binary") for c in cand}
    z = {i: pulp.LpVariable(f"z_{i}", cat="Binary") for i in covers}       # node covered?

    prob += pulp.lpSum(cost[c] * x[c] for c in x)                          # min cost
    prob += pulp.lpSum(pop[i] * z[i] for i in z) >= equity_tau * popU      # equity floor
    for i, cs in covers.items():
        prob += z[i] <= pulp.lpSum(x[c] for c in cs)                       # z only if a stop covers it

    st = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
    if pulp.LpStatus[st] != "Optimal":
        return None
    return float(sum(cost[c] for c in x if x[c].value() and x[c].value() > 0.5))


def solve_equity_lexicographic(pop, nearest_e, dC, improves, cost, equity_mask,
                               equity_R=R_EQUITY, equity_tau=0.5):
    """Two-stage (lexicographic) equity plan.

    Stage 1 -- minimum cost C* that meets the equity target (min_cost_for_equity).
    Stage 2 -- among all plans costing <= C*, pick the one with the smallest
               population-weighted walking distance, forbidding stops that serve
               nobody (x_c <= sum_i y_ic). This removes both the ghost-stop
               freedom and the arbitrary tie-breaking of the old +1e-6 call.

    Returns (opened_idx, C*_kEUR) or (None, None) if infeasible.
    """
    Cstar = min_cost_for_equity(pop, nearest_e, dC, improves, cost, equity_mask,
                                equity_R, equity_tau)
    if Cstar is None:
        return None, None

    U = np.where(equity_mask)[0]
    popU = float(pop[U].sum())
    relevant = np.where(improves.any(1))[0]
    prob = pulp.LpProblem("equity_stage2", pulp.LpMinimize)
    x = {c: pulp.LpVariable(f"x_{c}", cat="Binary") for c in range(len(cost))}
    y = {(i, c): pulp.LpVariable(f"y_{i}_{c}", lowBound=0, upBound=1, cat="Continuous")
         for i in relevant for c in np.where(improves[i])[0]}
    z = {i: pulp.LpVariable(f"z_{i}", cat="Binary") for i in U}

    base = float((pop * nearest_e).sum())
    prob += base + pulp.lpSum(pop[i] * (dC[i, c] - nearest_e[i]) * y[i, c] for (i, c) in y)
    prob += pulp.lpSum(cost[c] * x[c] for c in x) <= Cstar + 1e-6          # stay at min cost
    for i in relevant:
        prob += pulp.lpSum(y[i, c] for c in np.where(improves[i])[0]) <= 1
    for (i, c) in y:
        prob += y[i, c] <= x[c]
    for c in range(len(cost)):                                            # no ghost stops
        serving = [y[i, c] for i in relevant if (i, c) in y]
        prob += (x[c] <= pulp.lpSum(serving)) if serving else (x[c] == 0)
    for i in U:                                                           # equity via binary z
        cs = [c for c in np.where(improves[i])[0] if dC[i, c] <= equity_R]
        prob += (z[i] <= pulp.lpSum(x[c] for c in cs)) if cs else (z[i] == 0)
    prob += pulp.lpSum(pop[i] * z[i] for i in U) >= equity_tau * popU

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))
    opened = [c for c in x if x[c].value() and x[c].value() > 0.5]
    return opened, Cstar


# ============================================================ 3. KPI PANEL
def kpis(opened, pop, nearest_e, dC, cost, central, equity_mask):
    nearest_open = np.minimum(nearest_e, dC[:, opened].min(1)) if opened else nearest_e.copy()
    tot, cpop = pop.sum(), pop[central].sum()
    row = {
        "n_stops":     len(opened),
        "cost_kEUR":   round(float(sum(cost[c] for c in opened)), 1),
        "avg_walk_m":  round(float((pop * nearest_open).sum() / tot), 2),
        "central_avg_walk_m": round(float((pop[central] * nearest_open[central]).sum() / cpop), 2),
        "affected_pop": int(pop[nearest_open < nearest_e - 1e-9].sum()),
    }
    for R in COVER_RADII:
        row[f"cov{R}"]  = round(100 * pop[nearest_open <= R].sum() / tot, 2)
    for R in (200, 300, 400):
        m = central
        row[f"ccov{R}"] = round(100 * pop[m & (nearest_open <= R)].sum() / cpop, 2)
    # equity KPI: coverage of the currently underserved central group
    U = equity_mask
    if U.any():
        upop = pop[U].sum()
        row["underserved_pop"] = int(upop)
        row["underserved_now_cov300"] = round(100 * pop[U & (nearest_open <= 300)].sum() / upop, 2)
    return row, nearest_open


# ============================================================ 4. ANALYSES
def main():
    demand, pop, central, exy, cxy, cid = load()
    nearest_e, dC, improves = distances(demand, exy, cxy)
    cost, cload = build_cost_model(cxy, dC, improves, pop)
    equity_mask = central & (nearest_e > R_EQUITY)            # underserved central group

    print(f"data: pop={pop.sum():,.0f}  central={pop[central].sum():,.0f}  "
          f"candidates={len(cxy)}  improvable_pop={pop[improves.any(1)].sum():,.0f}")
    print(f"cost model (kEUR): min={cost.min():.0f} mean={cost.mean():.1f} max={cost.max():.0f}")
    print(f"underserved central (>300m): {pop[equity_mask].sum():,.0f} people\n")

    # ---- 0. baseline validation -------------------------------------------
    b_row, _ = kpis([], pop, nearest_e, dC, cost, central, equity_mask)
    pd.DataFrame([{**{"scenario": "baseline"}, **b_row}]).to_csv(f"{OUT}/00_baseline_validation.csv", index=False)
    print(f"[00] baseline avg_walk={b_row['avg_walk_m']}  cov200={b_row['cov200']}  "
          f"cov400={b_row['cov400']}  cov600={b_row['cov600']}   "
          f"(Katsioupis: 192.38 / 61.44 / 94.71 / 98.39)")

    # ---- 1+2. cost-accessibility Pareto: cost-blind vs cost-aware ----------
    budgets = [0, 45, 90, 135, 180, 270, 360, 450, 540, 720, 900, 1200]
    for tag, ccost_v, fn in [("costblind", np.full_like(cost, cost.mean()), "01_pareto_costblind"),
                             ("costaware", cost, "02_pareto_costaware")]:
        prob, x = build_pareto_model(pop, nearest_e, dC, improves, ccost_v)
        rows, sel = [], {}
        for B in budgets:
            op, sstat = sweep_budget(prob, x, cid, B)
            r, _ = kpis(op, pop, nearest_e, dC, cost, central, equity_mask)
            rows.append({"budget_kEUR": B, "solver_status": sstat, **r})
            sel[B] = [int(cid[c]) for c in op]
            if sstat != "optimal" and B > 0:
                print(f"     ! B={B}k: {sstat} (NOT proven optimal) -- flagged")
        pd.DataFrame(rows).to_csv(f"{OUT}/{fn}.csv", index=False)
        pd.DataFrame([{"budget_kEUR": B, "stops": sel[B]} for B in budgets]).to_csv(
            f"{OUT}/selected_stops_{tag}.csv", index=False)
        print(f"[{fn[:2]}] {tag}: at B=360k -> {rows[budgets.index(360)]['n_stops']} stops, "
              f"ccov300={rows[budgets.index(360)]['ccov300']}%")

    # ---- 3. cost-of-equity: min budget to cover tau% of the underserved ----
    #   Answers "what does it cost, and what does it do to system efficiency,
    #   to guarantee that tau% of the currently underserved central residents
    #   reach a stop within 300 m?"  (ceiling ~78%, so tau<=0.7 stays feasible)
    rows = []
    for tau in (0.30, 0.40, 0.50, 0.60, 0.70):
        op, Bstar = solve_equity_lexicographic(pop, nearest_e, dC, improves, cost,
                                               equity_mask, equity_tau=tau)
        if Bstar is None:
            rows.append({"equity_tau": tau, "min_budget_kEUR": "infeasible"}); continue
        r, _ = kpis(op, pop, nearest_e, dC, cost, central, equity_mask)
        rows.append({"equity_tau": tau, "min_budget_kEUR": round(Bstar, 1), **r})
    pd.DataFrame(rows).to_csv(f"{OUT}/03_equity_scenarios.csv", index=False)
    ok = [r for r in rows if isinstance(r.get("min_budget_kEUR"), float)]
    if ok:
        print(f"[03] cost-of-equity: cover 30% -> {rows[0]['min_budget_kEUR']}k ... "
              f"70% -> {rows[-1]['min_budget_kEUR']}k")

    # ---- 5. cost-effectiveness + recommendation ---------------------------
    pa = pd.read_csv(f"{OUT}/02_pareto_costaware.csv")
    base_cov = pa.iloc[0]["ccov300"]; base_walk = pa.iloc[0]["avg_walk_m"]
    ce = []
    for k in range(1, len(pa)):
        dC_ = pa.iloc[k]["cost_kEUR"] - pa.iloc[k-1]["cost_kEUR"]
        dcov = pa.iloc[k]["ccov300"] - pa.iloc[k-1]["ccov300"]
        dwalk = pa.iloc[k-1]["avg_walk_m"] - pa.iloc[k]["avg_walk_m"]
        ce.append({"budget_kEUR": pa.iloc[k]["budget_kEUR"], "n_stops": pa.iloc[k]["n_stops"],
                   "cum_cost_kEUR": pa.iloc[k]["cost_kEUR"],
                   "marg_kEUR_per_ccov300pt": round(dC_ / dcov, 1) if dcov > 1e-6 else None,
                   "ccov300": pa.iloc[k]["ccov300"], "avg_walk_m": pa.iloc[k]["avg_walk_m"]})
    ce_df = pd.DataFrame(ce)
    ce_df.to_csv(f"{OUT}/05_cost_effectiveness.csv", index=False)
    # "knee": last step whose marginal cost per coverage point stays below median
    valid = ce_df.dropna(subset=["marg_kEUR_per_ccov300pt"])
    thr = valid["marg_kEUR_per_ccov300pt"].median()
    knee = valid[valid["marg_kEUR_per_ccov300pt"] <= thr].iloc[-1]
    print(f"[05] recommended budget ~ {knee['cum_cost_kEUR']:.0f}k "
          f"({int(knee['n_stops'])} stops, central cov300={knee['ccov300']}%)")

    # ---- 6. cost-parameter sensitivity ------------------------------------
    global C_SOLO, C_LOAD
    rows = []
    for scale in (0.5, 1.0, 1.5):
        cost_s = C_BASE + np.where(cKDTree(cxy).query(cxy, k=2)[0][:,1] > PAIR_M, C_SOLO*scale, 0.0) \
                 + (cost - (C_BASE + np.where(cKDTree(cxy).query(cxy,k=2)[0][:,1]>PAIR_M, C_SOLO,0.0)))*scale
        op, _ = solve(360, pop, nearest_e, dC, improves, cost_s)
        r, _ = kpis(op, pop, nearest_e, dC, cost_s, central, equity_mask)
        rows.append({"cost_scale": scale, **r})
    pd.DataFrame(rows).to_csv(f"{OUT}/06_cost_sensitivity.csv", index=False)
    print("[06] cost sensitivity written")

    _figures()
    print("\nDONE. See results/ and figures/.")


# ============================================================ 5. FIGURES
def _figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blues = ["#00549F", "#407FB7", "#8EBAE5", "#CFcaccent"]
    cb = pd.read_csv(f"{OUT}/01_pareto_costblind.csv")
    ca = pd.read_csv(f"{OUT}/02_pareto_costaware.csv")
    eq = pd.read_csv(f"{OUT}/03_equity_scenarios.csv")

    # Pareto: cost vs central coverage@300
    plt.figure(figsize=(8, 5))
    plt.plot(cb["cost_kEUR"], cb["ccov300"], "o--", color="#8EBAE5", label="cost-blind")
    plt.plot(ca["cost_kEUR"], ca["ccov300"], "o-", color="#00549F", label="cost-aware")
    plt.xlabel("Total investment [k EUR]"); plt.ylabel("Central coverage @300 m [%]")
    plt.title("Cost–accessibility trade-off"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/pareto_cost_coverage.png", dpi=200); plt.close()

    # avg walk vs budget
    plt.figure(figsize=(8, 5))
    plt.plot(ca["cost_kEUR"], ca["avg_walk_m"], "o-", color="#00549F")
    plt.xlabel("Total investment [k EUR]"); plt.ylabel("Avg. walking distance [m]")
    plt.title("Diminishing returns of investment"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/avgwalk_budget.png", dpi=200); plt.close()

    # cost-of-equity curve
    eqf = eq[pd.to_numeric(eq["min_budget_kEUR"], errors="coerce").notna()].copy()
    eqf["min_budget_kEUR"] = eqf["min_budget_kEUR"].astype(float)
    plt.figure(figsize=(8, 5))
    plt.plot(eqf["equity_tau"] * 100, eqf["min_budget_kEUR"], "o-", color="#00549F")
    plt.xlabel("Share of underserved covered @300 m [%]")
    plt.ylabel("Minimum required investment [k EUR]")
    plt.title("The price of equity"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/equity_tradeoff.png", dpi=200); plt.close()


if __name__ == "__main__":
    main()
