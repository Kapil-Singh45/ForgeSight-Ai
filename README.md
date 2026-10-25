# 🔬 ForgeSight AI — Industrial Anomaly Detection & QA Inspection

ForgeSight AI is a production-oriented industrial visual anomaly detection and predictive maintenance platform built on **Python**, **PyTorch**, **Scikit-learn**, and **Streamlit**.

It solves a core manufacturing challenge: **detecting rare, unforeseen defects using only normal (defect-free) parts for training (unsupervised one-class learning).**

---

## 🌟 Key Highlights

- **State-of-the-Art Baseline (PatchCore):** Uses multi-scale feature embeddings from intermediate CNN layers (`layer2` & `layer3` of ResNet-18) with patch context aggregation and $10\%$ coreset memory bank subsampling.
- **Genuine Benchmark Performance (MVTec AD `bottle`):**
  - **Image-level ROC-AUC:** **100.0%** (`1.0000`)
  - **Image-level F1-Score:** **100.0%** (`1.0000`)
  - **Pixel-level ROC-AUC:** **98.28%** (`0.9828`)
  - **Per-Region Overlap (AU-PRO @ 30% FPR):** **91.19%** (`0.9119`)
  - **Calibrated Rejection Threshold:** **`1.9485`**
- **Interactive QA Inspection Station (`app.py`):**
  - Real-time PASS / REJECT quality verdicts based on calibrated decision boundaries.
  - Interactive anomaly heatmaps (Jet, Turbo, Viridis, Magma) with segmented defect contours in red.
  - Built-in factory test suite with one-click verification of normal and defective parts.
  - Live production shift audit logging with CSV export.
- **Cross-Platform & Cloud Ready:** Runs seamlessly on **Windows**, **Linux**, **Google Colab (T4 GPU)**, and **Streamlit Community Cloud**.

---

## 📁 Repository Structure

```text
ForgeSight_AI/
│
├── app.py                        # Streamlit Industrial QA Inspection Station (Web UI)
├── requirements.txt              # Production Python dependencies
├── .gitignore                    # Git rules for datasets, caches, and environment files
├── README.md                     # Comprehensive documentation & deployment guide
│
├── configs/
│   └── mvtec_config.yaml         # Category, resolution, backbone, and hyperparameter configuration
│
├── data/
│   ├── download_mvtec.py         # Automated streaming downloader for MVTec AD from Hugging Face
│   ├── test_samples/             # Standalone sample parts for quick testing
│   └── test_suite/               # 13 verified normal, defective, and synthetic flaw test images
│
├── src/
│   ├── __init__.py
│   ├── dataset.py                # PyTorch Dataset & DataLoader for MVTec AD splits
│   ├── transforms.py             # ImageNet normalization, bicubic resize, and mask binarization
│   ├── model.py                  # PyTorch PatchFeatureExtractor (multi-layer CNN feature hooks)
│   ├── anomaly_detector.py       # PatchCore memory bank, coreset subsampling, k-NN distance scorer
│   ├── metrics.py                # Image ROC-AUC, Pixel ROC-AUC, and Per-Region Overlap (AU-PRO)
│   └── utils.py                  # Seed management, device detection, and diagnostic visualizer
│
├── scripts/
│   ├── run_train.py              # Fits normal memory bank and saves detector checkpoint
│   ├── run_evaluate.py           # Evaluates test set, calculates benchmark metrics, exports reports
│   └── prepare_test_suite.py     # Generates verified normal & defective test suite images
│
├── notebooks/
│   └── colab_runner.ipynb        # Ready-to-run Google Colab workflow with GPU acceleration
│
└── outputs/
    ├── checkpoints/              # Trained feature memory banks (e.g. bottle_patchcore.pt)
    ├── heatmaps/                 # Exported 4-panel diagnostic visual inspection overlays
    └── reports/                  # JSON evaluation benchmark summaries (bottle_evaluation.json)
```

---

## 🚀 Quickstart Guide (Local Setup)

### 1. Clone the Repository & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/ForgeSight_AI.git
cd ForgeSight_AI

# Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **GPU Acceleration (Optional):**  
> If you have an NVIDIA GPU on Windows, install PyTorch with CUDA:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

### 2. Launch the Interactive Inspection Station

Run the Streamlit application:

```bash
streamlit run app.py
```

Open your browser at **`http://localhost:8501`**:
- **Conveyor Queue Tab:** Browse precomputed diagnostic inspection panels for normal and broken samples.
- **Manual Upload Tab:** Click any button in the verified test suite (e.g. `Normal Bottle 01`, `Broken Large 01`, `Synthetic Scratch`) or drag-and-drop any image to run live inference and view real-time defect heatmaps.

---

### 3. Model Training & Benchmark Evaluation (Optional / Re-run)

If you wish to re-train the memory bank or evaluate on another category:

```bash
# 1. Download MVTec AD category (e.g. bottle)
python data/download_mvtec.py --category bottle

# 2. Fit PatchCore memory bank on normal training samples
python scripts/run_train.py --category bottle

# 3. Run full evaluation across test set (normal + defects)
python scripts/run_evaluate.py --category bottle --num_vis 10
```

---

## ☁️ Deploying to Streamlit Community Cloud (Free)

ForgeSight AI is optimized for 1-click deployment on Streamlit Community Cloud:

1. Push your repository to **GitHub** (see Git steps below).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account.
3. Click **"New app"**.
4. Configure the deployment settings:
   - **Repository:** `<your-username>/ForgeSight_AI`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy"**.
6. Streamlit Cloud will install `requirements.txt`, load `outputs/checkpoints/bottle_patchcore.pt`, and launch your web station with a public shareable URL!

---

## 📊 How PatchCore Anomaly Detection Works

In industrial quality assurance, defects are diverse and rarely available during model training. ForgeSight AI uses an **unsupervised one-class anomaly detection** strategy:

1. **Feature Extraction:** A frozen pre-trained CNN (ResNet-18) extracts rich representations from `layer2` (texture) and `layer3` (object structure).
2. **Patch Context Pooling:** Average pooling aggregates spatial context around each token.
3. **Coreset Subsampling:** Subsamples $10\%$ of representative normal patches to build a compact, fast memory bank $\mathcal{M}$.
4. **Distance-Based Scoring:** Test patches are scored by their Euclidean distance to nearest neighbors in $\mathcal{M}$.
5. **Localization Heatmaps:** Continuous patch distances are mapped back to spatial coordinates, upscaled to input resolution, and smoothed with a Gaussian filter to segment defective regions.

---

## 📝 License & Dataset Attribution

- **Dataset:** MVTec Anomaly Detection (MVTec AD) by Bergmann et al. (CC BY-NC-SA 4.0). Intended for research and educational benchmarking.
- **Project Code:** MIT License.
