# aachen-bus-optimization
# Optimizing Public Transport Coverage in Aachen — Cost-Aware & Equity-Conscious Bus Stop Location

Seminar paper code & data — Chair of Data and Business Analytics, RWTH Aachen University.

Extends the p-median baseline of Katsioupis (2026) with heterogeneous
construction costs, a euro budget, a binary equity coverage floor for
underserved residents, and a cost–benefit appraisal (Standardisierte
Bewertung / GVFG NKV). Implemented in both Python (PuLP/CBC) and GAMSPy;
the two implementations are verified to produce identical results.

## Files

```
demand.csv, existing.csv, candidates.csv   spatial inputs (paper Ch. 3)
pois_aachen.csv, pois_aachen.xlsx          18 geocoded schools/hospitals (Sec. 3.2)
distances_e_new.csv, distances_c_new.csv   cached distance matrices (auto-regenerated)

aachen_model.py          core model + budget/equity experiments (PuLP/CBC)
aachen_cba.py            appraisal layer: benefits, per-stop & plan NKV
aachen_final.py          POI analysis + integrated plan comparison
aachen_model_gamspy.py   GAMSPy implementation (Task III); run directly to
                         reproduce and cross-check the reference results

METHODOLOGY_AND_RESULTS.md   detailed formulation, parameters, and result tables
```

## Reproduce (Python, no GAMS licence needed)

```bash
pip install pandas numpy scipy pulp matplotlib openpyxl
python aachen_model.py      # budget & equity experiments
python aachen_cba.py        # cost–benefit appraisal
python aachen_final.py      # POI analysis + final comparison
```

## GAMSPy verification (Task III, requires a GAMS licence)

```bash
pip install gamspy pandas numpy scipy
python aachen_model_gamspy.py
```
Prints each result next to its reference value. Expected agreement:

```
base (status-quo Z) = 47,228,220.00
Model A  B=360k -> stops [1050, 1072, 1082, 1089]
Model A  B=450k -> stops [1050, 1072, 1082, 1089, 1093]
Equity plan (tau=0.5) -> C* = 450.0, stops [1050, 1065, 1089, 1092, 1123, 1159]
```
The GAMSPy and Python implementations encode the identical formulation
(paper Sec. 4.3–4.4: cost-budgeted p-median, binary equity coverage, and
the lexicographic equity plan) and were confirmed to return identical stop
selections on the full instance.

## Data sources

Existing stops: AVV Open Data Portal. Candidates: screening procedure of
Katsioupis (2026). Demand: Schug et al. (2021), 10 m census-disaggregated grid
aggregated to 50 m. POIs: OSM-identified facilities, coordinates geocoded from
official addresses. Cost levels: documented German municipal projects
(Arnsberg, Bayreuth, Euskirchen/NVR, LNVG). Appraisal parameters:
Standardisierte Bewertung 2016+ (VTTS 6.60 EUR/h), BVWP 2030 (discount 1.7%),
GVFG (NKV >= 1 funding criterion).
