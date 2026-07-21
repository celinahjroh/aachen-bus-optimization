"""
================================================================================
 Aachen Public Transport Coverage — FINAL PIPELINE (runs everything)
================================================================================
 Business Analytics Project, SoSe 2026.
 One entry point that produces every table and figure the paper needs:

   PART A  (aachen_model.py)   baseline validation, cost-blind vs cost-aware
                               Pareto, price of equity, target scenarios,
                               cost-effectiveness, cost sensitivity
   PART B  (aachen_cba.py)     monetised cost-benefit appraisal: per-stop NKV,
                               net-benefit-optimal plan, plan comparison,
                               CBA sensitivity
   PART C  (this file)         POI ACCESSIBILITY: access to schools & hospitals
                               (Meeting Summary: "access to schools, hospitals,
                               or vulnerable communities")
   PART D  (this file)         FINAL INTEGRATED PLAN COMPARISON: all planning
                               philosophies x all KPI families in one table

 POI DATA: pois_aachen.csv — 18 schools/hospitals, manually compiled from
 public map data (documented, ~100-200 m accuracy, fine for 300-400 m
 catchments). Replaceable 1:1 by an official OSM Overpass export
 (amenity=school / amenity=hospital); the pipeline reads any CSV with
 columns name,type,lat,lon.

 Run:   python aachen_final.py            (full run, ~10-15 min)
        python aachen_final.py --poi-only (parts C+D, needs A+B results)
================================================================================
"""
import os
import sys
import ast
import numpy as np
import pandas as pd
import pulp
from scipy.spatial import cKDTree

import aachen_model as A
import aachen_cba as CBA

OUT = "results"; FIG = "figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

R_POI_CATCH = 400          # residents within this walking distance of a POI = catchment
R_ACCESS    = 300          # "good access" walking threshold (AVV)


# ============================================================ POI machinery
def load_pois(path="pois_aachen.csv"):
    pois = pd.read_csv(path, comment="#")
    if not {"x", "y"}.issubset(pois.columns):
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        pois["x"], pois["y"] = tr.transform(pois["lon"].values, pois["lat"].values)
    return pois


def poi_masks(pois, demand, nearest_e):
    """catchment = residents within R_POI_CATCH walking of any POI;
       underserved catchment = catchment residents whose baseline stop walk > R_ACCESS."""
    d_poi = cKDTree(pois[["x", "y"]].to_numpy()).query(demand)[0] * A.F
    catch = d_poi <= R_POI_CATCH
    under = catch & (nearest_e > R_ACCESS)
    return catch, under, d_poi


def poi_stop_distances(pois, exy, cxy, opened_idx):
    """Walking distance from each POI to its nearest stop (existing + opened)."""
    stops = exy if not opened_idx else np.vstack([exy, cxy[opened_idx]])
    return cKDTree(stops).query(pois[["x", "y"]].to_numpy())[0] * A.F


# ============================================================ PART C: POI analysis
def part_c_poi(demand, pop, central, exy, cxy, cid, nearest_e, dC, improves, cost):
    pois = load_pois()
    catch, under, _ = poi_masks(pois, demand, nearest_e)
    equity_mask = central & (nearest_e > A.R_EQUITY)

    print(f"[C] POIs: {len(pois)} ({(pois.type=='hospital').sum()} hospitals, "
          f"{(pois.type=='school').sum()} schools)")
    print(f"[C] catchment pop (<= {R_POI_CATCH} m of a POI): {pop[catch].sum():,.0f}  "
          f"| currently underserved therein: {pop[under].sum():,.0f}")

    # ceiling: how much of the underserved catchment can candidates fix at all?
    fixable = under & (dC.min(1) <= R_ACCESS)
    ceiling = pop[fixable].sum() / pop[under].sum() if pop[under].sum() else 0
    print(f"[C] achievable ceiling for underserved catchment: {ceiling*100:.1f} %")

    # POI-priority plan: min cost to cover tau of the underserved catchment
    tau = round(min(0.5, max(0.1, 0.8 * ceiling)), 2)
    plan_poi, Bstar = A.solve_equity_lexicographic(pop, nearest_e, dC, improves, cost,
                                                   equity_mask=under, equity_R=R_ACCESS,
                                                   equity_tau=tau)
    plan_poi = plan_poi or []
    if Bstar:
        print(f"[C] POI-priority plan: cover {tau:.0%} of underserved catchment "
              f"for {Bstar:.0f}k EUR ({len(plan_poi)} stops)")

    # per-POI stop access, baseline vs plans
    rows = []
    for pname, op in [("baseline", []), ("poi_priority", plan_poi)]:
        d = poi_stop_distances(pois, exy, cxy, op)
        rows.append({"plan": pname,
                     "mean_poi_stop_walk_m": round(float(d.mean()), 1),
                     "max_poi_stop_walk_m": round(float(d.max()), 1),
                     "pois_within_200m_%": round(100 * (d <= 200).mean(), 1),
                     "pois_within_300m_%": round(100 * (d <= 300).mean(), 1)})
    poi_side = pd.DataFrame(rows)
    poi_side.to_csv(f"{OUT}/10_poi_stop_access.csv", index=False)

    # per-POI detail table (paper appendix)
    d0 = poi_stop_distances(pois, exy, cxy, [])
    detail = pois[["name", "type"]].copy()
    detail["stop_walk_m_baseline"] = d0.round(0)
    detail.sort_values("stop_walk_m_baseline", ascending=False).to_csv(
        f"{OUT}/10b_poi_detail.csv", index=False)
    print(f"[C] POI stop access (baseline): {poi_side.iloc[0]['pois_within_300m_%']}% "
          f"of POIs have a stop within 300 m; mean {poi_side.iloc[0]['mean_poi_stop_walk_m']} m")

    return pois, catch, under, plan_poi, tau, Bstar


# ============================================================ PART D: integrated comparison
def part_d_final(demand, pop, central, exy, cxy, cid, nearest_e, dC, improves, cost,
                 catch, under, plan_poi, tau_poi):
    equity_mask = central & (nearest_e > A.R_EQUITY)
    benefit = CBA.benefit_matrix(pop, nearest_e, dC, improves)
    id2idx = {int(cid[c]): c for c in range(len(cid))}

    plans = {"status_quo": []}
    sa = pd.read_csv(f"{OUT}/selected_stops_costaware.csv")
    plans["cost_aware_360k"] = [id2idx[s] for s in ast.literal_eval(
        sa[sa.budget_kEUR == 360].iloc[0]["stops"])]
    eq_plan, Beq = A.solve_equity_lexicographic(pop, nearest_e, dC, improves, cost,
                                                equity_mask, equity_tau=0.5)
    if Beq:
        plans["equity_first_50pct"] = eq_plan
    if plan_poi:
        plans[f"poi_priority_{int(tau_poi*100)}pct"] = plan_poi
    plans["net_benefit_opt"] = CBA.solve_netbenefit(pop, nearest_e, dC, improves, cost, benefit)

    rows = []
    for name, op in plans.items():
        k, nearest_open = A.kpis(op, pop, nearest_e, dC, cost, central, equity_mask)
        ap = CBA.plan_appraisal(op, pop, nearest_e, dC, cost, benefit)
        cpop = pop[catch].sum(); upop = pop[under].sum()
        rows.append({
            "plan": name, "n_stops": k["n_stops"], "cost_kEUR": k["cost_kEUR"],
            "NKV": ap["NKV"], "net_benefit_kEUR": ap["net_benefit_kEUR"],
            "avg_walk_m": k["avg_walk_m"], "central_cov300_%": k["ccov300"],
            "underserved_cov300_%": k.get("underserved_now_cov300"),
            "poi_catchment_cov300_%": round(100 * pop[catch & (nearest_open <= 300)].sum() / cpop, 2),
            "poi_underserved_cov300_%": round(100 * pop[under & (nearest_open <= 300)].sum() / upop, 2),
        })
    final = pd.DataFrame(rows)
    final.to_csv(f"{OUT}/12_final_plan_comparison.csv", index=False)
    print("\n[D] FINAL INTEGRATED PLAN COMPARISON:")
    print(final.to_string(index=False))

    # figure: who benefits under which philosophy
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = final.set_index("plan")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    idx = np.arange(len(f)); w = 0.28
    ax.bar(idx - w, f["central_cov300_%"], w, label="central pop", color="#00549F")
    ax.bar(idx,      f["underserved_cov300_%"].astype(float), w, label="underserved (central)", color="#E30066")
    ax.bar(idx + w,  f["poi_underserved_cov300_%"], w, label="underserved near schools/hospitals", color="#57AB27")
    ax.set_xticks(idx); ax.set_xticklabels(f.index, rotation=15, ha="right")
    ax.set_ylabel("Coverage @300 m [%]")
    ax.set_title("Same city, different priorities: who gets access under each plan")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(f"{FIG}/final_who_benefits.png", dpi=200); plt.close()
    print(f"[D] figure -> {FIG}/final_who_benefits.png")
    return final


# ============================================================ MAIN
def main(poi_only=False):
    if not poi_only:
        A.main()                                  # PART A
        CBA.main()                                # PART B
    demand, pop, central, exy, cxy, cid = A.load()
    nearest_e, dC, improves = A.distances(demand, exy, cxy)
    cost, _ = A.build_cost_model(cxy, dC, improves, pop)
    pois, catch, under, plan_poi, tau, Bstar = part_c_poi(
        demand, pop, central, exy, cxy, cid, nearest_e, dC, improves, cost)
    part_d_final(demand, pop, central, exy, cxy, cid, nearest_e, dC, improves, cost,
                 catch, under, plan_poi, tau)
    print("\nALL DONE — results/ has tables 00–12, figures/ has all plots.")


if __name__ == "__main__":
    main(poi_only="--poi-only" in sys.argv)
