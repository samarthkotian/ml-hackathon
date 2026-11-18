# Evaluation corrections

The historical artifacts committed before this branch used several different
evaluation rules. They remain available for traceability, not as validation of
the corrected implementation. The saved classifiers have not been retrained.

## Current protocol: `wta-lcs-v2`

- Candidate F1 always uses the highest-probability candidate per observation.
  Neither evaluation labels nor the better of two prediction rules select the rule.
- Tuning reports the actual fold means without a post-hoc offset. The baseline
  tuning trial and winning trial use the same subsample and folds. The seven-model
  full-validation benchmark is a different CV population.
- Classifier and HMM continuity use the same `HMMConfig.max_bfs_hops` limit.
- Route sequence precision/recall/F1 count longest-common-subsequence matches,
  preserving order and repeated segments. `route_set_f1` and `route_jaccard`
  separately describe unordered overlap. Old `route_seq_f1` values were set F1.
- The API accepts safe trip names only, confines file access to configured
  directories, requires an explicit `train` partition for training-trip lookup,
  and does not return reference-route geometry or local filesystem paths.
- The UI discards its previous response before a request and whenever the server,
  trip or partition changes. A failed request leaves no stale download or metrics.

## Reproduce corrected measurements

Install the project requirements and place the original training data in
`Data/Student folder/train/<trip_id>/`. The ten student-test trips are included.
The evaluator partition must be supplied separately in one of the locations in
`src/config.py`; it is not included in this repository.

```powershell
python train.py
python src/phase2_experiments.py
python predict.py --partition student_test
python predict.py --partition evaluator_test
```

These commands intentionally replace generated results when the required datasets
are available. New prediction rows identify the evaluation protocol and hop limit.
The hardcoded Phase 1 baseline in the Phase 2 comparison remains historical; it
cannot be reconstructed with the current Phase 2 feature implementation.
Full cross-validation and evaluator results have not been rerun in this fix.

The saved Phase 2 classifier was run on all ten included student-test trips after
the fix. `outputs/results/student_test_corrected_v2.csv` contains these corrected
measurements: mean trip top-1 accuracy 0.9446525558, mean ordered LCS F1
0.9140508870 (the historical set-overlap F1 was 0.9473500521).
All source dataset files were checked before and after inference and were unchanged.
The validation environment used scikit-learn 1.9.1, LightGBM 4.7.0 and XGBoost
3.4.1. Loading the existing pickle emitted version-compatibility warnings
(scikit-learn training version 1.9.0 and an older XGBoost serialization); these
student-test top-1 scores nevertheless match the previously saved results.

Run the focused regression suite with development dependencies:

```powershell
pip install -r requirements.txt pytest httpx
python -m pytest tests -q
```

## Deployment evidence

The old terminal/dashboard PNGs are renamed `*_illustration.png`; the old log is
`illustrative_api_log.txt`. They were drawn using local function calls and a mock
client address. The historical PDFs contain those illustrations and must not be
used as proof of a two-laptop deployment.

Start Uvicorn on Laptop 1, then run this command on Laptop 2:

```powershell
python src/capture_deployment_screenshots.py --server-url http://<LAPTOP_1_IP>:8000
```

Despite its legacy filename, the script now makes actual HTTP requests and saves
the observed response to `outputs/deployment/http_verification.json`. It fails on
HTTP/network errors and does not fabricate screenshots, addresses or access logs.
The record verifies the HTTP endpoints from the machine running the command; it
cannot establish that the machine is a second physical laptop or that Streamlit
rendered correctly. Open Streamlit on Laptop 2 and take real screenshots separately
if deployment evidence is needed.

The report generator refuses legacy metric inputs. Regenerate the training and
Phase 2 artifacts before publishing an updated report. Historical PDFs are kept
under filenames ending in `_historical.pdf`.
