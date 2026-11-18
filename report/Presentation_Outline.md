# PathSnap: Phase 1 & Phase 2 Pitch & Viva Defense Guide — Team SI No 107

> Historical presentation: the scores below predate the `wta-lcs-v2` corrections.
> Old sequence F1 is unordered set overlap, and the included deployment figures
> are illustrations. See `docs/EVALUATION_CORRECTIONS.md` before using these claims.

**Course:** UE24CS352A: Machine Learning — PES University (RR Campus)  
**Team Members:**
1. **PES1UG24CS395 — Saatvik Gupta** (Data Architecture, Projection, 18m Spatial-Chord Bearing & Candidate Search)
2. **PES1UG24CS396 — Sagar A Walikar** (7-Model Benchmarking, `GroupKFold` CV, Bayesian Tuning & Phase 2 GBDT Ensemble)
3. **PES1UG24CS406 — Samarth Kotian** (Road-Graph HMM, U-Turn-Aware Directional Log-Viterbi Decoder & Path Bridging)
4. **PES1UG24CS407 — Samarth M H** (Two-Laptop FastAPI Server, Streamlit Evaluator UI & Folium Maps)

---

## Part 1: What We Learned & Improved in Phase 2 (Key Talking Points for the Panel)

### 1. Diagnosis of Phase 1 Residual Errors (Without Touching `evaluator_test/`)
- In Phase 1, our pipeline achieved **`91.13%` Observation-Level Top-1 Accuracy** on `student_test` (`92.20%` trip-mean) and **`89.68%` Observation-Level Top-1 Accuracy** (`92.75%` trip-mean) on `evaluator_test`.
- Inspecting our worst development trip (`student_test/00000077`, `79.75%` accuracy) revealed a striking finding: **81% of all remaining errors (`16.39%` out of `20.25%`) were Reverse-Twin Arcs (`u -> v` vs `v -> u`)** on undivided two-way streets!
- **Why did Reverse-Twin errors happen?**
  1. On a two-way street, directed arcs `u -> v` and `v -> u` share the exact same geometry and therefore have **identical perpendicular distance $d_{\perp}$** to the GPS point. Sorting candidates by distance alone broke ties arbitrarily by `arc_id`.
  2. When a vehicle stops at traffic lights or creeps in slow congestion (such as `evaluator_test/00000015` with 6,854 GPS points over 136 segments = ~50s per segment), a fixed 1-second or 5-second time window measures stationary GPS jitter rather than road direction.
  3. In our Phase 1 HMM, slow backward drift (`-0.6 m` to `-4.0 m/s`) on a reverse-twin arc was not penalized.

### 2. Three Leakage-Free Phase 2 Upgrades
1. **18-Meter Spatial-Chord Travel Bearing (`src/data_loader.py`):** Instead of a fixed time window, for every GPS observation $t$ we find the nearest point before $t$ and after $t$ where the car has displaced by at least **18 meters in space**. Even if a car sits at a red light for 60 seconds, the 18m spatial chord points in the true travel direction of the street with **99.5% median cosine agreement**.
2. **Sub-Millimeter Twin-Arc Tie-Breaking & U-Turn-Aware Directional HMM (`src/candidates.py`, `src/hmm_viterbi.py`):**
   - Breaks distance ties between twin arcs (`u -> v` vs `v -> u`) using $10^{-4}\text{ m} \times (1 - \cos \Delta \theta_{\text{chord}})$.
   - Adds spatial-chord directional alignment to the HMM emission, tightens backward-progress detection on directed arcs to $\Delta \text{prog} < -0.6\text{ m}$, and penalizes immediate U-turn flips (`u -> v` to `v -> u`).
3. **Soft-Voting GBDT Ensemble (`0.60 Bayesian-Tuned LightGBM + 0.40 Regularised XGBoost`):**
   - Combines leaf-wise histogram GBDT (`LightGBM`) and depth-wise GBDT (`XGBoost`), trained strictly on the 80 training trips (`RANDOM_SEED = 107`).

### 3. Phase 1 vs Phase 2 Score Jump

| Metric | `student_test` (Phase 1) | `student_test` (Phase 2) | **Gain** | `evaluator_test` (Phase 1 Baseline) | `evaluator_test` (Phase 2 Final) | **Gain** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Observation-Level Top-1 Accuracy** | `91.13%` | **`94.96%`** | **`+3.83%`** | `89.68%` | **`94.66%`** (`94.76%` Exp 3) | **`+4.98%`** (nearly 50% error reduction!) |
| **Mean Trip Top-1 Accuracy** | `92.20%` | **`94.47%`** | **`+2.27%`** | `92.75%` | **`94.57%`** (`94.63%` Exp 3) | **`+1.82%`** |
| **Mean Candidate Macro F1-Score** | `0.9568` | **`0.9696`** | **`+0.0128`** | `0.9607` | **`0.9710`** | **`+0.0103`** |
| **Topological Route Continuity** | `99.66%` | **`99.62%`** | — | `99.57%` | **`99.62%`** | **`+0.05%`** |
| **Mean Sequence Route Overlap F1** | `0.9394` | **`0.9474`** | **`+0.0080`** | `0.9082` | **`0.9294`** | **`+0.0212`** |
| **Hardest Trip `00000015` (6,854 pts)** | — | — | — | `80.80%` (`70.56%` Clf) | **`93.84%`** (`93.14%` Clf) | **`+13.04%`** |
| **Hardest Dev Trip `00000077` (3,911 pts)** | `79.75%` | **`95.47%`** | **`+15.72%`** | — | — | — |

---

## Part 2: One-Paragraph Phase 2 Summary to Speak to the Panel

> **"When Phase 2 started, we first wired the 10 reserved `evaluator_test` trips into our read-only evaluation pipeline and established a Phase 1 baseline of 89.68% observation-level Top-1 accuracy (92.75% trip average) and 99.57% route continuity. Next, without ever training or tuning on `evaluator_test`, we analyzed our errors on the `student_test` development set and discovered that 81% of remaining mistakes were 'reverse-twin' errors on undivided two-way streets—where both directions of the same street have identical perpendicular distance, and stopping at traffic lights makes short-window GPS headings noisy. We solved this by computing an 18-meter spatial-chord bearing that looks 18 meters backward and forward in space so red-light stops never corrupt vehicle direction, added sub-millimeter twin-arc tie-breaking and U-turn penalties in our Viterbi decoder, and ensembled our Bayesian-tuned LightGBM with XGBoost. This cut our error rate nearly in half: `student_test` accuracy jumped from 91.13% to 94.96%, and held-out `evaluator_test` observation accuracy jumped from 89.68% to 94.66% (with the 6,854-point urban traffic trip `00000015` jumping from 80.80% to 93.84% and `00000067` achieving 100.0% sequence route F1)."**
