# PathSnap: Distributed Map-Matching & Route Reconstruction
**Course:** UE24CS352A — Machine Learning | PES University (RR Campus)  
**Team 11**

---

## 🗺️ Project Overview

Raw GPS tracking data is inherently noisy due to urban multipath reflections, signal attenuation, canyon effects, and low sampling rates. Vehicles often appear to veer off roads, jump across city blocks, or navigate impassable terrain. 

**PathSnap** is a distributed, end-to-end machine learning system designed to solve map-matching and reconstruct coherent vehicle trajectories by mapping noisy GPS observations onto real-world road networks. The system blends:
1. **Geometric & Spatial Feature Engineering:** Metric projection, KDTree spatial indexing, perpendicular distance, and directional alignment.
2. **Supervised Candidate Classification:** Multi-model experimentation comparing 5 distinct algorithm families, optimized via **Bayesian Hyperparameter Search (Optuna)**.
3. **Sequential Route Reconstruction (HMM + Viterbi):** Combining classifier emission likelihoods with road-network topological transition probabilities and GPS gap handling.
4. **Distributed Deployment:** A decoupled two-laptop client-server architecture powered by a high-throughput **FastAPI** backend and an interactive **Streamlit + Folium** frontend.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│               LAPTOP 2: STREAMLIT CLIENT                │
│  - Trip Selector (10 Reserved Evaluation Trips)         │
│  - Interactive Folium Map (GPS / Arcs / Routes)         │
│  - Real-Time Scorecard (Accuracy, Continuity, Latency)   │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP / REST (LAN)
                           ▼
┌─────────────────────────────────────────────────────────┐
│               LAPTOP 1: FASTAPI BACKEND                 │
│  - GET  /health           : Health & model status       │
│  - GET  /trips            : List evaluation trip IDs    │
│  - GET  /predict/{trip_id}: Complete inference pipeline │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 1. Local Metric Projection & Spatial Feature Eng. │  │
│  │ 2. Tuned HistGradientBoosting Candidate Scoring   │  │
│  │ 3. NetworkX Road Graph & Transition Modeling     │  │
│  │ 4. Log-Space Viterbi Sequential Decoding          │  │
│  │ 5. Server-Side Metric Computation (Zero Leakage)  │  │
│  └───────────────────────────────────────────────────┘  │
│  Internal Storage: Reference Routes (Server-side only)  │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
ML-Hackathon-Team11/
├── artifacts/
│   ├── bayesian_opt/          # Optuna trials, best_params.json, study DB, tuned vs untuned
│   ├── models/                # Saved weights for 5 trained models + tuned model
│   ├── plots/
│   │   ├── folium_maps/       # Pre-rendered interactive HTML maps for test trips
│   │   └── sample_trip_folium.html
│   └── results/               # Metric CSVs (model_comparison.csv, viterbi_evaluation.csv)
├── notebooks/
│   ├── 01_EDA_and_Visualization.ipynb
│   ├── 02_Candidate_Generation_and_Features.ipynb
│   └── 03_Model_Training_and_Comparison.ipynb
├── Phase_2/
│   ├── 04_Phase2_BayesianOpt_HMM_Viterbi_Maps.ipynb
│   ├── backend/
│   │   └── app.py             # Laptop 1: FastAPI server
│   ├── bayesian_opt/
│   │   └── bayesian_tune.py   # Optuna Bayesian optimization pipeline
│   ├── frontend/
│   │   └── app.py             # Laptop 2: Streamlit client interface
│   ├── hmm/
│   │   ├── emissions.py       # Classifier + Gaussian distance emission modeling
│   │   ├── transitions.py     # Graph-based shortest path transition modeling
│   │   └── viterbi.py         # Log-space Viterbi dynamic programming decoder
│   ├── viz/
│   │   └── folium_maps.py     # Automated Folium map generation
│   └── run_phase2.py          # Master runner for Phase 2
├── src/
│   ├── data/
│   │   ├── loader.py          # Data ingestion (.track, .nodes, .arcs, .route)
│   │   └── inventory.py       # Dataset audit & partition inspection
│   ├── features/
│   │   ├── spatial.py         # Local transverse Mercator projector & geometry
│   │   ├── candidates.py      # KDTree spatial indexing of road arcs
│   │   ├── alignment.py       # Ground-truth trajectory alignment
│   │   └── build_features.py  # End-to-end feature extraction pipeline
│   └── models/
│       ├── metrics.py         # Top-1 Accuracy, Macro-F1, Continuity, Log Loss
│       └── train_comparison.py # Multi-model training and evaluation harness
├── Student folder/            # Dataset (train / student_test partitions)
├── requirements.txt           # Project dependencies
└── README.md
```

---

## ⚙️ Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Arjun-Gowda/ML-Hackathon-Team11.git
cd ML-Hackathon-Team11
```

### 2. Set Up a Virtual Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Two-Laptop Deployment Guide (Live Evaluation)

The system is configured to run distributed across two machines connected to the same Local Area Network (Wi-Fi or Hotspot).

### Step 1: Start the Backend on Laptop 1 (Model Host)
1. Find Laptop 1's local IP address:
   ```bash
   # On Linux/macOS:
   ip route get 1.1.1.1 | awk '{print $7}'
   # On Windows:
   ipconfig
   ```
2. Start the FastAPI server (bind to `0.0.0.0` on port 8000):
   ```bash
   uvicorn Phase_2.backend.app:app --host 0.0.0.0 --port 8000
   ```
3. Verify server health locally or in browser:
   ```bash
   curl http://localhost:8000/health
   ```

### Step 2: Launch the Frontend on Laptop 2 (Client UI)
1. On Laptop 2, clone the repository and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit application:
   ```bash
   streamlit run Phase_2/frontend/app.py
   ```
3. In the sidebar, set the **Laptop 1 Server URL** (e.g. `http://192.168.1.15:8000` replacing with Laptop 1's LAN IP).
4. Click **🔄 Check Server Health**.
5. Select any of the 10 evaluation trips from the dropdown and click **🚀 Run Map-Matching**.

---

## 🔬 Experimental Pipeline & Results

### 1. Multi-Model Comparison (Candidate Classification)
Evaluated across 80 training trips (GroupKFold by trip) and evaluated on 10 student testing trips:

| Algorithm Family | Model | CV Macro F1 | CV Top-1 Acc | Student-Test Top-1 Acc | Student-Test Macro F1 | Student-Test Log Loss | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear / Probabilistic** | Logistic Regression | 0.7538 | 0.6266 | 0.8027 | 0.7639 | 0.2976 | < 0.01 ms |
| **Instance-Based** | Weighted KNN ($k=15$) | 0.8194 | 0.6203 | 0.7968 | 0.8734 | 0.2282 | 2.68 ms |
| **Bagged Tree Ensemble** | Random Forest ($n=100$) | 0.8061 | 0.6381 | 0.8164 | 0.8574 | 0.1806 | 0.02 ms |
| **Gradient Boosting** | **HistGradientBoosting** | **0.8077** | **0.6389** | **0.8187** | **0.8495** | **0.1926** | **0.02 ms** |
| **Neural Network** | Shallow MLP (64, 32) | 0.8341 | 0.6343 | 0.8109 | 0.8870 | 0.1211 | 0.02 ms |

*Selected Model:* **HistGradientBoosting** — selected for its high Top-1 candidate ranking accuracy, well-calibrated posterior probabilities, and near-instantaneous inference latency.

### 2. Bayesian Hyperparameter Optimization
Using **Optuna** (50 trials with Group 5-Fold Cross-Validation):
- **Tuned Hyperparameters:** `max_iter`, `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`.
- **Top-1 Accuracy:** Improved from **81.87%** (default) to **81.96%** (tuned).
- **Macro F1-Score:** Improved from **0.8495** to **0.8566**.
- **Log Loss:** Reduced from **0.1926** to **0.1768**.

### 3. Sequential Route Reconstruction (HMM + Viterbi)
Comparing classifier-only predictions with the HMM + Viterbi decoder across the 10 student-test trips:

- **Mean Top-1 Accuracy:** Improved from **83.87%** $\rightarrow$ **86.31%** (+2.44% gain).
- **Mean Route Continuity:** Improved from **89.62%** $\rightarrow$ **96.66%** (+7.04% gain).
- **Dropout & Gap Handling:** Robust relaxation of transition constraints during temporal gaps ($\Delta t > 5.0$ s).

---

## 🔒 Evaluation Integrity & Anti-Leakage Compliance
- **Zero Label Leakage:** Ground truth route files are strictly retained server-side on Laptop 1 for evaluation metric calculation and never returned to Laptop 2.
- **Dynamic Inference:** Every prediction is computed on the fly through the complete pipeline (projection $\rightarrow$ candidate retrieval $\rightarrow$ classifier scoring $\rightarrow$ Viterbi trellis). No hardcoded responses exist.
