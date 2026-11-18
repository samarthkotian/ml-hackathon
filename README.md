# PathSnap: Distributed Map-Matching & Route Reconstruction (Phase 1 & Phase 2 Complete)

> **Evaluation correction:** This branch uses `wta-lcs-v2`: fixed winner-takes-all
> candidate scoring, raw cross-validation scores, a shared continuity hop limit,
> and order-sensitive LCS route F1. Existing benchmark tables, plots, saved CSVs,
> model metadata and PDFs are historical results from the earlier protocol;
> their F1/CV/continuity values must not be presented as corrected results.
> Student-test top-1 accuracies were independently checked and match the supplied
> predictions. See [the correction notes](docs/EVALUATION_CORRECTIONS.md) for scope,
> reproduction steps and deployment evidence requirements.

**PES University — Department of Computer Science & Engineering (RR Campus)**  
**Course:** UE24CS352A: Machine Learning Hackathon

### Team: SI No 107
| SRN | Name | Role & Module Ownership |
| :--- | :--- | :--- |
| **PES1UG24CS395** | **Saatvik Gupta** | Data Architecture, Local Projection, 18m Spatial-Chord Bearing & Multi-Point `cKDTree` Candidate Search |
| **PES1UG24CS396** | **Sagar A Walikar** | 7-Model Benchmarking, `GroupKFold` CV, Optuna TPE Bayesian Tuning & Phase 2 Soft-Voting GBDT Ensemble |
| **PES1UG24CS406** | **Samarth Kotian** | Road-Graph HMM Formulation, Directional U-Turn-Aware Log-Viterbi Decoder & Path Bridging |
| **PES1UG24CS407** | **Samarth M H** | Two-Laptop Deployment (`FastAPI` Backend on Laptop 1, `Streamlit` Evaluator UI on Laptop 2 & Folium Maps) |

---

## 1. Problem Overview & Phase 1 $\rightarrow$ Phase 2 Technical Evolution

Urban GPS trajectories suffer from multipath reflections, building occlusion, signal dropouts (up to $220\text{ s}$ in `evaluator_test/00000073`), and bidirectional street ambiguities. **PathSnap** reconstructs topologically coherent vehicle routes across a directed road network graph (`.nodes`, `.arcs`) from noisy 1-Hz GPS observations (`.track`):

1. **Local Tangent-Plane Metric Projection (`src/data_loader.py`):**
   - Projects angular WGS84 `(longitude, latitude)` into local metric coordinates $(x, y)$ in meters centered at each trip's centroid ($\bar{\lambda}, \bar{\phi}$), eliminating the $\cos(\phi)$ longitudinal distortion ($\sim 35\%$ distortion at $50^\circ\text{N}$).
   - Aligns chronological `.track` GPS observations with the compressed ground-truth `.route` sequence using monotonic dynamic programming.
2. **Multi-Point Segment `cKDTree` Candidate Generation (`src/candidates.py`):**
   - Indexes 5 control points ($0\%, 25\%, 50\%, 75\%, 100\%$) along every directed road segment in a 2D `scipy.spatial.cKDTree` and ranks the top $K = 8$ closest unique road segments by exact perpendicular point-to-segment distance, achieving a **0.52% Candidate Miss Rate** on `student_test` and **0.78% Candidate Miss Rate** on `evaluator_test`.
3. **Phase 2 Breakthrough — 18m Spatial-Chord Bearing & Twin-Arc Tie-Breaking (`src/data_loader.py`, `src/candidates.py`, `src/features.py`):**
   - *Root Cause Discovered on Development Set:* Error analysis on development trip `student_test/00000077` revealed that **81% of residual errors** occurred on undivided two-way streets where opposite-direction twin arcs (`u -> v` and `v -> u`) have the **exact same perpendicular distance** ($d_{\perp}(u\to v) \equiv d_{\perp}(v\to u)$) and low-speed traffic stops ($v_t < 2\text{ m/s}$) corrupt fixed-time-window GPS headings.
   - *Solution:* We compute a **stationary-robust 18m spatial-chord travel bearing** $\theta_{\text{chord}, t}$ (looking backward and forward along the trajectory until the vehicle has displaced by $\ge 18\text{ m}$ in space), use sub-millimeter ($10^{-4}\text{ m}$) directional tie-breaking between twin arcs, and stabilize low-speed instantaneous headings.
4. **Phase 2 Soft-Voting GBDT Ensemble & Directional U-Turn-Aware HMM/Viterbi (`src/phase2_experiments.py`, `src/hmm_viterbi.py`):**
   - Combines our two strongest gradient-boosted families (**Bayesian-Tuned LightGBM** weight `0.60` + **Regularised XGBoost** weight `0.40`, trained strictly on the 80 training trips with `RANDOM_SEED = 107`) with a directional HMM emission term, strict backward-progress detection ($\Delta \text{prog} < -0.6\text{ m}$), and immediate U-turn transition penalties.

---

## 2. Experimental Results Summary

### 2.1 Phase 1 vs Phase 2 Pipeline Progression (`student_test` & Held-Out `evaluator_test`)

| Experiment Stage | Description | `student_test` Mean Top-1 | `student_test` Obs Top-1 | `student_test` Macro F1 | `evaluator_test` Mean Top-1 | `evaluator_test` Obs Top-1 | `evaluator_test` Macro F1 | `evaluator_test` Continuity | `evaluator_test` Seq Route F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1 (Phase 1 Baseline)** | Phase 1 Tuned LightGBM + 5-Step Heading + Phase 1 HMM | `92.20%` | `91.13%` | `0.9568` | `92.75%` | `89.68%` | `0.9607` | `99.57%` | `0.9082` |
| **Exp 2 (+Spatial-Chord & Dir-HMM)** | Phase 1 Classifier + 18m Spatial-Chord Twin Tie-Break + Directional HMM | `94.05%` | `94.54%` | `0.9673` | `94.50%` | `94.48%` | `0.9706` | `99.59%` | `0.9275` |
| **Exp 3 (Retrained P2 LightGBM)** | Retrained Bayesian-Tuned LightGBM on Phase 2 Features + Directional HMM | `94.39%` | `94.86%` | `0.9692` | **`94.63%`** | **`94.76%`** | **`0.9714`** | **`99.62%`** | **`0.9294`** |
| **Exp 4 (P2 LightGBM+XGB Ensemble)** | **Selected Final Model:** `0.6 Tuned LightGBM + 0.4 XGBoost` + Directional HMM | **`94.47%`** | **`94.96%`** (`+3.83%`) | **`0.9696`** | `94.57%` | `94.66%` (`+4.98%`) | `0.9710` | **`99.62%`** | **`0.9294`** |

---

### 2.2 Final Held-Out `evaluator_test/` Results (10 Reserved Evaluation Trips — Read-Only Verification)

| Trip ID | GPS Obs | Road Arcs | Pred Route Arcs | Cand Miss Rate | Phase 1 HMM Top-1 | Phase 2 Clf-Only Top-1 | **Phase 2 HMM Top-1** | **Phase 1 $\rightarrow$ 2 Gain** | Candidate Macro F1 | HMM Route Continuity | Sequence Route F1 | Latency (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00000015` | 6,854 | 3,225 | 339 | `0.13%` | `80.80%` | `93.14%` | **`93.84%`** | **`+13.04%`** | `0.9652` | `100.00%` | `0.7941` | `740.9 ms` |
| `00000017` | 1,346 | 76,425 | 835 | `2.08%` | `92.42%` | `91.75%` | **`92.12%`** | `-0.30%` | `0.9606` | `99.26%` | `0.9550` | `388.4 ms` |
| `00000023` | 458 | 10,596 | 100 | `2.62%` | `91.70%` | `92.79%` | **`91.48%`** | `-0.22%` | `0.9584` | `98.69%` | `0.7436` | `97.3 ms` |
| `00000030` | 654 | 24,218 | 339 | `1.22%` | `83.49%` | `81.96%` | **`87.00%`** | **`+3.52%`** | `0.9289` | `99.54%` | `0.9196` | `164.4 ms` |
| `00000049` | 2,642 | 47,542 | 424 | `0.26%` | `99.05%` | `98.79%` | **`99.02%`** | `-0.04%` | `0.9951` | `99.70%` | `0.9844` | `549.9 ms` |
| `00000059` | 1,031 | 24,024 | 488 | `1.16%` | `95.25%` | `94.47%` | **`94.67%`** | `-0.58%` | `0.9727` | `99.32%` | `0.9382` | `212.4 ms` |
| `00000067` | 916 | 4,485 | 82 | `0.00%` | `97.27%` | `97.82%` | **`98.47%`** | **`+1.20%`** | `0.9913` | `100.00%` | **`1.0000`** | `156.7 ms` |
| `00000073` | 1,144 | 25,631 | 249 | `0.00%` | `98.86%` | `98.78%` | **`98.34%`** | `-0.52%` | `0.9905` | `99.91%` | `0.9916` | `219.1 ms` |
| `00000087` | 1,399 | 107,451 | 456 | `0.07%` | `99.36%` | `95.28%` | **`99.29%`** | `-0.07%` | `0.9961` | `99.86%` | `0.9978` | `370.5 ms` |
| `00000094` | 3,230 | 14,931 | 680 | `0.12%` | `89.29%` | `92.04%` | **`91.49%`** | **`+2.20%`** | `0.9517` | `99.94%` | `0.9694` | `547.0 ms` |
| **AGGREGATE** | **19,674** | **33,853** | **399.2** | **`0.77%`** | `89.68%` obs (`92.75%` mean) | `93.68%` mean | **`94.66%` obs (`94.57%` mean)** | **`+4.98%` obs (`+1.82%` mean)** | **`0.9710`** | **`99.62%`** | **`0.9294`** | **`344.7 ms`** |

---

### 2.3 Multi-Family Model Comparison (7 Models Across 6 Algorithm Families on 80 Train Trips 5-Fold `GroupKFold` CV)

| Model | Algorithm Family | CV Macro F1 | Student-Test Top-1 Acc | Pointwise Route Continuity | Student-Test Macro F1 | Log Loss | Inference Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | `0.8831 ± 0.0549` | `85.03%` | `98.17%` | `0.9158` | `0.2153` | `6.8 ms/trip` |
| **Distance-Weighted KNN** | Instance-Based | `0.8945 ± 0.0450` | `85.36%` | `97.77%` | `0.9176` | `0.1437` | `808.6 ms/trip` |
| **Extra Trees** | Tree-Based (Randomised) | `0.8929 ± 0.0683` | `86.70%` | `97.74%` | `0.9253` | `0.1408` | `20.4 ms/trip` |
| **Random Forest** | Ensemble (Bagging) | `0.8961 ± 0.0651` | `87.71%` | `98.23%` | `0.9311` | `0.1077` | `16.7 ms/trip` |
| **MLP Neural Network** | Neural Networks | `0.9069 ± 0.0445` | `88.14%` | `98.24%` | `0.9336` | `0.0816` | `24.4 ms/trip` |
| **XGBoost** | Gradient Boosting (Depth-wise) | `0.9165 ± 0.0333` | `88.39%` | `98.51%` | `0.9350` | `0.0742` | `7.6 ms/trip` |
| **LightGBM** | Gradient Boosting (Histogram) | `0.9149 ± 0.0351` | `88.48%` | `98.43%` | `0.9355` | `0.0741` | `21.5 ms/trip` |

---

## 3. Quickstart & One-Command Execution

### 3.1 Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3.2 Run Standalone Inference & Evaluator Verification (`predict.py`)
```powershell
python predict.py
# Or evaluate only the 10 reserved evaluator_test trips:
python predict.py --partition evaluator_test --generate_maps
```

### 3.3 Run Complete Training Pipeline (`train.py`) & Phase 2 Experiments (`src/phase2_experiments.py`)
```powershell
python train.py
python src/phase2_experiments.py
```

### 3.4 Launch Full Stack on One Laptop (`run_local_demo.py`)
```powershell
python run_local_demo.py
```

### 3.5 Two-Laptop Live Evaluation
- **On Laptop 1 (FastAPI Backend Server):**
  ```powershell
  uvicorn api.main:app --host 0.0.0.0 --port 8000
  ```
- **On Laptop 2 (Streamlit Evaluator UI):**
  ```powershell
  streamlit run ui/app.py
  ```
  *(Set the FastAPI URL in the left sidebar to `http://<LAPTOP_1_IP>:8000`; the dropdown automatically defaults to the 10 reserved `evaluator_test` trips).*
