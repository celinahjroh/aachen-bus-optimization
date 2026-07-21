"""
================================================================================
 Aachen Public Transport Coverage — GAMSPy IMPLEMENTATION (handout deliverable)
 Cost-Budgeted p-Median with Binary Equity Coverage + Lexicographic Equity Plan
================================================================================
 This file encodes, in GAMSPy, EXACTLY the same formulation as the current
 aachen_model.py (the pure-Python implementation that generates every table
 and figure in the paper). It is provided to satisfy Task III of the handout
 ("implement your model using GAMSPy") and to let the supervisor verify, with
 a GAMS licence, that the two implementations agree.

 It mirrors three functions from aachen_model.py one-to-one:
   build_pareto_model() / sweep_budget()  -> solve_budget()      (Model A)
   min_cost_for_equity()                  -> min_cost_for_equity_gamspy()  (Model B)
   solve_equity_lexicographic()           -> solve_equity_lexicographic_gamspy()
                                              (Model B, stage 1, then Model C, stage 2)

 Data preprocessing (loading, distance computation, dominance sparsification,
 and the heterogeneous cost model) is IDENTICAL code to aachen_model.py --
 duplicated here in plain NumPy so this file is self-contained and does not
 import aachen_model.py. Both files must therefore always agree on preprocessing;
 if aachen_model.py's load()/distances()/build_cost_model() ever change, this
 file's copies below must be updated to match.

 ------------------------------------------------------------------------------
 STATUS: this file has not yet been executed. Development happened without a
 GAMS licence (no .gamspy_venv available in this environment). Before citing
 "GAMSPy reproduces the Python results" in the paper, run this file once in
 the group's licensed environment and compare against the reference values
 embedded as comments below (captured from the current aachen_model.py):

   base (status-quo Z)            = 47,228,220.00   (-> avg walk 192.38 m)
   total population                = 245,489.37
   relevant (improvable) nodes     = 850
   underserved central population  = 6,032.18
   budget sweep B=360k  -> stops {1050, 1072, 1082, 1089}          (optimal)
   budget sweep B=450k  -> stops {1050, 1072, 1082, 1089, 1093}    (optimal)
   equity tau=0.5       -> C* = 450.0k, stops {1050,1065,1089,1092,1123,1159}
 ------------------------------------------------------------------------------

 Run in the group's GAMSPy venv, with existing.csv / candidates.csv / demand.csv
 in the working directory:
     C:\\Users\\pierr\\.gamspy_venv\\Scripts\\python.exe aachen_model_gamspy.py
================================================================================
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum, ModelStatus

# ------------------------------------------------------------------ constants
# Identical to aachen_model.py -- keep these two files in sync.
F           = 1.3
R_EQUITY    = 300
C_BASE      = 45.0
C_SOLO      = 15.0
C_LOAD      = (0.0, 20.0, 40.0)
PAIR_M      = 40.0


# ============================================================ 1. DATA + DISTANCES
# (identical logic to aachen_model.py's load() / distances() / build_cost_model();
#  duplicated here in plain NumPy so this file is standalone)
def load():
    ex  = pd.read_csv("existing.csv")
    ca  = pd.read_csv("candidates.csv")
    dem = pd.read_csv("demand.csv")
    demand  = dem[["x", "y"]].to_numpy(float)
    pop     = dem["population"].to_numpy(float)
    central = (dem["central"].to_numpy() == 1)
    exy = ex[["x", "y"]].to_numpy(float)
    cxy = ca[["x", "y"]].to_numpy(float)
    cid = ca["stop_id"].to_numpy()
    did = dem["demand_id"].to_numpy()
    return demand, pop, central, exy, cxy, cid, did


def distances(demand, exy, cxy):
    nearest_e = cKDTree(exy).query(demand)[0] * F
    dC = np.sqrt(((demand[:, None, :] - cxy[None, :, :]) ** 2).sum(-1)) * F
    improves = dC < nearest_e[:, None] - 1e-9
    return nearest_e, dC, improves


def build_cost_model(cxy, dC, improves, pop):
    nn2 = cKDTree(cxy).query(cxy, k=2)[0][:, 1]
    solo = nn2 > PAIR_M
    load = np.array([pop[improves[:, c]].sum() for c in range(len(cxy))])
    tier = np.zeros(len(cxy), int)
    pos = load[load > 0]
    if len(pos):
        q1, q2 = np.quantile(pos, [1 / 3, 2 / 3])
        tier = np.where(load <= q1, 0, np.where(load <= q2, 1, 2))
    cost = C_BASE + np.where(solo, C_SOLO, 0.0) + np.array([C_LOAD[t] for t in tier])
    return cost, load


# ============================================================ 2. GAMSPy BUILDING BLOCKS
def _sparse_records(demand, pop, nearest_e, dC, improves, cid, did):
    """Build the (i, j_c) admissible-pair records used by every GAMSPy model
    below -- the GAMSPy equivalent of aachen_model.py's `relevant` node list
    and the dominance-sparsified pairs. Only these ~5,197 pairs are given to
    GAMSPy; this is what keeps the MIP the same size as the PuLP version."""
    relevant = np.where(improves.any(1))[0]
    i_ids = [str(did[i]) for i in relevant]
    j_ids = [str(cid[c]) for c in range(len(cid))]

    pop_recs = [(str(did[i]), float(pop[i])) for i in relevant]
    ne_recs  = [(str(did[i]), float(nearest_e[i])) for i in relevant]

    dC_recs, A_recs = [], []
    for i in relevant:
        for c in np.where(improves[i])[0]:
            dC_recs.append((str(did[i]), str(cid[c]), float(dC[i, c])))
            A_recs.append((str(did[i]), str(cid[c]), 1))

    return relevant, i_ids, j_ids, pop_recs, ne_recs, dC_recs, A_recs


def _equity_records(pop, dC, improves, cid, did, equity_mask, equity_R=R_EQUITY):
    """(demand_id, population) for the equity target group, plus the sparse
    (i, j_c) coverage-within-R pairs used by the binary z_i variable."""
    U = np.where(equity_mask)[0]
    u_ids  = [str(did[i]) for i in U]
    u_pop  = [(str(did[i]), float(pop[i])) for i in U]
    cov_recs = []
    for i in U:
        for c in np.where(improves[i])[0]:
            if dC[i, c] <= equity_R:
                cov_recs.append((str(did[i]), str(cid[c]), 1))
    return u_ids, u_pop, cov_recs


# ============================================================ 3. MODEL A: cost-budgeted p-median
def solve_budget(budget, pop, nearest_e, dC, improves, cost, cid, did):
    """GAMSPy equivalent of aachen_model.solve()/sweep_budget(): cost-budgeted
    p-median, tau=0 (no equity requirement). One fresh Container per call --
    simplest and safest given the moderate problem size (141 candidates,
    ~5,197 assignment pairs); mirrors PuLP's fresh LpProblem() per solve.

    Returns (opened_stop_ids, status_string).
    """
    relevant, i_ids, j_ids, pop_recs, ne_recs, dC_recs, A_recs = _sparse_records(
        None, pop, nearest_e, dC, improves, cid, did)

    m = Container()
    i   = Set(m, "i", records=i_ids)
    j_c = Set(m, "j_c", records=j_ids)

    pop_p   = Parameter(m, "pop_p", domain=[i], records=pop_recs)
    ne_p    = Parameter(m, "ne_p", domain=[i], records=ne_recs)
    dC_p    = Parameter(m, "dC_p", domain=[i, j_c], records=dC_recs)
    A_p     = Parameter(m, "A_p", domain=[i, j_c], records=A_recs)
    cost_p  = Parameter(m, "cost_p", domain=[j_c],
                        records=list(zip([str(s) for s in cid], cost)))
    base    = float((pop * nearest_e).sum())          # constant, computed once in NumPy

    x = Variable(m, "x", domain=[j_c], type="binary")
    y = Variable(m, "y", domain=[i, j_c], type="positive")
    y.up[i, j_c] = 1

    assign = Equation(m, "assign", domain=[i])
    assign[i] = Sum(j_c, y[i, j_c]) <= 1

    link = Equation(m, "link", domain=[i, j_c])
    link[i, j_c] = y[i, j_c] <= x[j_c]

    admiss = Equation(m, "admiss", domain=[i, j_c])
    admiss[i, j_c] = y[i, j_c] <= A_p[i, j_c]

    budget_eq = Equation(m, "budget_eq")
    budget_eq[...] = Sum(j_c, cost_p[j_c] * x[j_c]) <= budget

    noghost = Equation(m, "noghost", domain=[j_c])
    noghost[j_c] = x[j_c] <= Sum(i, y[i, j_c])

    model = Model(m, "aachen_budget",
                  equations=[assign, link, admiss, budget_eq, noghost],
                  problem="mip", sense="min",
                  objective=base + Sum((i, j_c),
                            pop_p[i] * (dC_p[i, j_c] - ne_p[i]) * y[i, j_c]))
    model.solve()

    if model.status not in (ModelStatus.OptimalGlobal, ModelStatus.Integer):
        return [], str(model.status)
    chosen = x.records[x.records["level"] > 0.5]["j_c"].astype(int).tolist()
    status = "optimal" if model.status == ModelStatus.OptimalGlobal else "feasible"
    return sorted(chosen), status


# ============================================================ 4. MODEL B: min cost for equity target
def min_cost_for_equity_gamspy(pop, nearest_e, dC, improves, cost, equity_mask, cid, did,
                               equity_R=R_EQUITY, equity_tau=0.5):
    """GAMSPy equivalent of aachen_model.min_cost_for_equity(). Note this model
    has NO assignment variables y at all -- like the Python version, it only
    needs to decide which stops to open (x) and whether each underserved node
    ends up covered (binary z), because the objective is cost, not distance.

    Returns C* in kEUR, or None if infeasible.
    """
    j_ids = [str(s) for s in cid]
    u_ids, u_pop_recs, cov_recs = _equity_records(pop, dC, improves, cid, did,
                                                   equity_mask, equity_R)
    if not cov_recs:
        return None

    m = Container()
    j_c = Set(m, "j_c", records=j_ids)
    u   = Set(m, "u", records=u_ids)

    cost_p = Parameter(m, "cost_p", domain=[j_c], records=list(zip(j_ids, cost)))
    upop_p = Parameter(m, "upop_p", domain=[u], records=u_pop_recs)
    cov_p  = Parameter(m, "cov_p", domain=[u, j_c], records=cov_recs)
    popU   = float(sum(v for _, v in u_pop_recs))

    x = Variable(m, "x", domain=[j_c], type="binary")
    z = Variable(m, "z", domain=[u], type="binary")

    zcov = Equation(m, "zcov", domain=[u])
    zcov[u] = z[u] <= Sum(j_c, cov_p[u, j_c] * x[j_c])

    equity_eq = Equation(m, "equity_eq")
    equity_eq[...] = Sum(u, upop_p[u] * z[u]) >= equity_tau * popU

    model = Model(m, "mincost_equity", equations=[zcov, equity_eq],
                  problem="mip", sense="min",
                  objective=Sum(j_c, cost_p[j_c] * x[j_c]))
    model.solve()

    if model.status not in (ModelStatus.OptimalGlobal, ModelStatus.Integer):
        return None
    return float(model.objective_value)


# ============================================================ 5. MODEL C: lexicographic stage 2
def solve_equity_lexicographic_gamspy(pop, nearest_e, dC, improves, cost, equity_mask,
                                      cid, did, equity_R=R_EQUITY, equity_tau=0.5):
    """GAMSPy equivalent of aachen_model.solve_equity_lexicographic(). Stage 1
    calls min_cost_for_equity_gamspy() for C*; stage 2 fixes the budget at C*
    and minimises walking distance, with the same no-ghost-stop guarantee and
    binary z-coverage as the Python version.

    Returns (opened_stop_ids, C*_kEUR) or (None, None) if infeasible.
    """
    Cstar = min_cost_for_equity_gamspy(pop, nearest_e, dC, improves, cost,
                                       equity_mask, cid, did, equity_R, equity_tau)
    if Cstar is None:
        return None, None

    relevant, i_ids, j_ids, pop_recs, ne_recs, dC_recs, A_recs = _sparse_records(
        None, pop, nearest_e, dC, improves, cid, did)
    u_ids, u_pop_recs, cov_recs = _equity_records(pop, dC, improves, cid, did,
                                                   equity_mask, equity_R)
    base = float((pop * nearest_e).sum())
    popU = float(sum(v for _, v in u_pop_recs))

    m = Container()
    i   = Set(m, "i", records=i_ids)
    j_c = Set(m, "j_c", records=j_ids)
    u   = Set(m, "u", records=u_ids)                     # equity group (standalone set)

    pop_p  = Parameter(m, "pop_p", domain=[i], records=pop_recs)
    ne_p   = Parameter(m, "ne_p", domain=[i], records=ne_recs)
    dC_p   = Parameter(m, "dC_p", domain=[i, j_c], records=dC_recs)
    A_p    = Parameter(m, "A_p", domain=[i, j_c], records=A_recs)
    cost_p = Parameter(m, "cost_p", domain=[j_c], records=list(zip(j_ids, cost)))
    upop_p = Parameter(m, "upop_p", domain=[u], records=u_pop_recs)
    cov_p  = Parameter(m, "cov_p", domain=[u, j_c], records=cov_recs)

    x = Variable(m, "x", domain=[j_c], type="binary")
    y = Variable(m, "y", domain=[i, j_c], type="positive")
    y.up[i, j_c] = 1
    z = Variable(m, "z", domain=[u], type="binary")

    assign = Equation(m, "assign", domain=[i]); assign[i] = Sum(j_c, y[i, j_c]) <= 1
    link   = Equation(m, "link", domain=[i, j_c]); link[i, j_c] = y[i, j_c] <= x[j_c]
    admiss = Equation(m, "admiss", domain=[i, j_c]); admiss[i, j_c] = y[i, j_c] <= A_p[i, j_c]

    budget_eq = Equation(m, "budget_eq")
    budget_eq[...] = Sum(j_c, cost_p[j_c] * x[j_c]) <= Cstar + 1e-6   # stay at min cost (stage 1 result)

    noghost = Equation(m, "noghost", domain=[j_c])
    noghost[j_c] = x[j_c] <= Sum(i, y[i, j_c])

    zcov = Equation(m, "zcov", domain=[u])
    zcov[u] = z[u] <= Sum(j_c, cov_p[u, j_c] * x[j_c])

    equity_eq = Equation(m, "equity_eq")
    equity_eq[...] = Sum(u, upop_p[u] * z[u]) >= equity_tau * popU

    model = Model(m, "equity_stage2",
                  equations=[assign, link, admiss, budget_eq, noghost, zcov, equity_eq],
                  problem="mip", sense="min",
                  objective=base + Sum((i, j_c),
                            pop_p[i] * (dC_p[i, j_c] - ne_p[i]) * y[i, j_c]))
    model.solve()

    if model.status not in (ModelStatus.OptimalGlobal, ModelStatus.Integer):
        return None, Cstar
    chosen = x.records[x.records["level"] > 0.5]["j_c"].astype(int).tolist()
    return sorted(chosen), Cstar


# ============================================================ 6. DEMO / VERIFICATION RUN
def main():
    demand, pop, central, exy, cxy, cid, did = load()
    nearest_e, dC, improves = distances(demand, exy, cxy)
    cost, load_v = build_cost_model(cxy, dC, improves, pop)
    equity_mask = central & (nearest_e > R_EQUITY)

    print(f"base (status-quo Z) = {(pop * nearest_e).sum():,.2f}  "
          f"(reference: 47,228,220.00)")
    print(f"total pop = {pop.sum():,.2f}  (reference: 245,489.37)")
    print(f"underserved = {pop[equity_mask].sum():,.2f}  (reference: 6,032.18)\n")

    print("Budget sweep (Model A):")
    for B in (360, 450):
        stops, status = solve_budget(B, pop, nearest_e, dC, improves, cost, cid, did)
        print(f"  B={B}k -> {status}, stops={[int(s) for s in stops]}")
    print("  reference: B=360 -> optimal {1050,1072,1082,1089} | "
          "B=450 -> optimal {1050,1072,1082,1089,1093}\n")

    print("Lexicographic equity plan, tau=0.5 (Models B+C):")
    stops, Cstar = solve_equity_lexicographic_gamspy(
        pop, nearest_e, dC, improves, cost, equity_mask, cid, did, equity_tau=0.5)
    print(f"  C* = {Cstar}, stops={[int(s) for s in stops] if stops else None}")
    print("  reference: C* = 450.0, stops={1050,1065,1089,1092,1123,1159}")


if __name__ == "__main__":
    main()
