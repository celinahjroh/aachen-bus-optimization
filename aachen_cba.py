"""
================================================================================
 Aachen Bus Stops — COST-BENEFIT LAYER (Standardisierte-Bewertung style)
================================================================================
"""
import os
import numpy as np
import pandas as pd
import pulp
import aachen_model as A            

OUT = "results"; os.makedirs(OUT, exist_ok=True)

# ---- appraisal parameters (all citable / transparent; see sensitivity) --------
VTTS_EUR_H   = 6.60      # value of travel time, Standardisierte Bewertung Formblatt 20
WALK_MS      = 1.25      # walking speed 4.5 km/h
TRIPS_YEAR   = 150.0     # home-end transit-access walks per resident per year (~0.4/day)
LIFE_YEARS   = 30        # asset evaluation period
DISCOUNT     = 0.017     # discount rate (BVWP 2030)
PV_FACTOR    = (1 - (1 + DISCOUNT) ** -LIFE_YEARS) / DISCOUNT      # ~23.3


def benefit_matrix(pop, nearest_e, dC, improves,
                   vtts=VTTS_EUR_H, trips=TRIPS_YEAR, pv=PV_FACTOR):
    saved_m = np.where(improves, nearest_e[:, None] - dC, 0.0)          
    hours = saved_m / WALK_MS / 3600.0
    b = pop[:, None] * trips * hours * vtts * pv / 1000.0               
    return np.where(improves, b, 0.0)


def solve_netbenefit(pop, nearest_e, dC, improves, cost, benefit):
    relevant = np.where(improves.any(1))[0]
    prob = pulp.LpProblem("netbenefit", pulp.LpMaximize)
    x = {c: pulp.LpVariable(f"x_{c}", cat="Binary") for c in range(len(cost))}
    y = {(i, c): pulp.LpVariable(f"y_{i}_{c}", lowBound=0, upBound=1, cat="Continuous")
         for i in relevant for c in np.where(improves[i])[0]}
    prob += (pulp.lpSum(benefit[i, c] * y[i, c] for (i, c) in y)
             - pulp.lpSum(cost[c] * x[c] for c in x))
    for i in relevant:
        prob += pulp.lpSum(y[i, c] for c in np.where(improves[i])[0]) <= 1
    for (i, c) in y:
        prob += y[i, c] <= x[c]
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))
    return [c for c in x if x[c].value() and x[c].value() > 0.5]


def plan_appraisal(opened, pop, nearest_e, dC, cost, benefit):
    if not opened:
        return {"n_stops": 0, "cost_kEUR": 0.0, "benefit_kEUR": 0.0, "net_benefit_kEUR": 0.0, "NKV": None}
    sub = dC[:, opened]
    best = sub.min(1); arg = np.array(opened)[sub.argmin(1)]
    served = best < nearest_e - 1e-9
    saved_m = np.where(served, nearest_e - best, 0.0)
    ben = (pop * TRIPS_YEAR * (saved_m / WALK_MS / 3600.0) * VTTS_EUR_H * PV_FACTOR).sum() / 1000.0
    cst = float(sum(cost[c] for c in opened))
    return {"n_stops": len(opened), "cost_kEUR": round(cst, 1),
            "benefit_kEUR": round(ben, 1), "net_benefit_kEUR": round(ben - cst, 1),
            "NKV": round(ben / cst, 2) if cst > 0 else None}


def main():
    demand, pop, central, exy, cxy, cid = A.load()
    nearest_e, dC, improves = A.distances(demand, exy, cxy)
    cost, _ = A.build_cost_model(cxy, dC, improves, pop)
    benefit = benefit_matrix(pop, nearest_e, dC, improves)
    equity_mask = central & (nearest_e > A.R_EQUITY)

    print(f"PV factor={PV_FACTOR:.1f}  VTTS={VTTS_EUR_H} EUR/h  trips/yr={TRIPS_YEAR}")

    op_nb = solve_netbenefit(pop, nearest_e, dC, improves, cost, benefit)
    ap_nb = plan_appraisal(op_nb, pop, nearest_e, dC, cost, benefit)
    print(f"[NB] net-benefit-optimal: {ap_nb['n_stops']} stops, cost {ap_nb['cost_kEUR']}k, "
          f"benefit {ap_nb['benefit_kEUR']}k, NKV={ap_nb['NKV']}")

    rows = []
    for c in np.where(improves.any(0))[0]:
        ap = plan_appraisal([c], pop, nearest_e, dC, cost, benefit)
        rows.append({"stop_id": int(cid[c]), "cost_kEUR": ap["cost_kEUR"],
                     "benefit_kEUR": ap["benefit_kEUR"], "NKV": ap["NKV"]})
    per_stop = pd.DataFrame(rows).sort_values("NKV", ascending=False)
    per_stop.to_csv(f"{OUT}/07_per_stop_NKV.csv", index=False)
    worth = (per_stop["NKV"] >= 1.0).sum()
    print(f"[07] {worth}/{len(per_stop)} candidate stops are individually fundable (NKV>=1)")

    plans = {}
    sa = pd.read_csv(f"{OUT}/selected_stops_costaware.csv")
    import ast
    id2idx = {int(cid[c]): c for c in range(len(cid))}
    plan_costaware = [id2idx[s] for s in ast.literal_eval(sa.iloc[6]["stops"])]   # ~B=360k
    plans["cost_aware_360k"] = plan_costaware
    plans["net_benefit_opt"] = op_nb
    op_eq, Bstar = A.solve_equity_lexicographic(pop, nearest_e, dC, improves, cost,
                                                equity_mask, equity_tau=0.5)
    if Bstar:
        plans["equity_first_50pct"] = op_eq

    comp = []
    for name, op in plans.items():
        ap = plan_appraisal(op, pop, nearest_e, dC, cost, benefit)
        k, _ = A.kpis(op, pop, nearest_e, dC, cost, central, equity_mask)
        comp.append({"plan": name, **ap, "central_cov300": k["ccov300"],
                     "avg_walk_m": k["avg_walk_m"],
                     "underserved_cov300": k.get("underserved_now_cov300")})
    comp_df = pd.DataFrame(comp)
    comp_df.to_csv(f"{OUT}/08_plan_comparison_CBA.csv", index=False)
    print("[08] plan comparison (with NKV) written")

    rows = []
    for trips in (75, 150, 300):
        for disc in (0.01, 0.017, 0.03):
            pv = (1 - (1 + disc) ** -LIFE_YEARS) / disc
            b = benefit_matrix(pop, nearest_e, dC, improves, trips=trips, pv=pv)
            op = solve_netbenefit(pop, nearest_e, dC, improves, cost, b)
            # appraise with these params
            sub = dC[:, op]; best = sub.min(1); served = best < nearest_e - 1e-9
            saved = np.where(served, nearest_e - best, 0.0)
            ben = (pop * trips * (saved / WALK_MS / 3600) * VTTS_EUR_H * pv).sum() / 1000
            cst = float(sum(cost[c] for c in op))
            rows.append({"trips_year": trips, "discount": disc, "n_stops": len(op),
                         "cost_kEUR": round(cst, 1), "benefit_kEUR": round(ben, 1),
                         "NKV": round(ben / cst, 2) if cst else None})
    pd.DataFrame(rows).to_csv(f"{OUT}/09_CBA_sensitivity.csv", index=False)
    print("[09] CBA sensitivity written")

    print("\nDONE (CBA). See results/07_*, 08_*, 09_*.")


if __name__ == "__main__":
    main()
