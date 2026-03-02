# Zomato Kitchen Prep Time (KPT) Prediction - Solution

**Zomathon 2026 - Problem Statement 1**

## Team
**Authors:** Atrishman Mukherjee, Ansh Mathur, Supratik Dey

## Problem Statement
Improving Kitchen Prep Time (KPT) Prediction to Optimize Rider Assignment and Customer ETA at Zomato

## Project Structure

```
Zomathon/
├── data/                       # Synthetic datasets (merchants, orders, kitchen rush, IoT sensors)
├── src/                       # Python scripts
│   ├── generate_data.py       # Data generation
│   ├── analyze_kpt.py         # KPT analysis
│   └── generate_visualizations.py  # Visualization generation
├── notebooks/                  # Jupyter notebooks for analysis
├── images/                    # Generated visualizations
├── latex/                     # LaTeX source and compiled PDF
│   ├── main.tex              # Main LaTeX document
│   └── main.pdf              # Compiled report (31 pages)
├── models/                    # Trained models and results
├── train_kpt_standalone.py    # Self-contained model training script
├── codebase.py               # Utility functions and model definitions
└── README.md
```

## Solution Overview

### The Problem
- Current KPT prediction relies on biased merchant-marked signals
- Only 44% visibility into total kitchen load (missing other platforms, dine-in)
- Current system costs ₹262M+ annually in inefficiencies

### Our Approach
**Multi-Signal Fusion Framework** combining:
1. IoT-based kitchen load sensors (thermal, motion, ambient)
2. Enhanced merchant app with proactive prompts
3. ML pattern detection and prediction
4. External API integration (weather, events)
5. Adaptive signal weighting based on reliability

### Key Innovation
**5 New Engineered Features:**
1. Order Value per Item (complexity indicator)
2. Distance to Nearest Rush Hour (temporal dynamics)
3. Kitchen Efficiency Score (load-to-capacity ratio)
4. Order Complexity Score (normalized item/value metric)
5. Merchant Historical Bias (systematic error tracking) - **Rank #2 feature**

## Results

### Model Performance
Trained on 50,000 orders from 1,000 merchants:

| Metric | Baseline | Improved | Change |
|--------|----------|----------|---------|
| MAE | 3.30 min | 2.32 min | **-29.9%** |
| RMSE | 4.30 min | 3.08 min | **-28.3%** |
| R² Score | 0.9482 | 0.9733 | **+2.7%** |
| P50 Error | 2.59 min | 1.78 min | **-31.3%** |
| P90 Error | 7.10 min | 5.01 min | **-29.5%** |
| P95 Error | 8.96 min | 6.34 min | **-29.2%** |

### Business Impact
- **Annual Savings:** ₹2.7 Billion (scaled to 1.5M daily orders)
- **Breakeven:** 2-3 months
- **5-Year ROI:** 13,400%+

## Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Quick Start

**Option 1: Run everything**
```bash
python run_all.py
```

**Option 2: Step-by-step**

1. Generate synthetic datasets:
   ```bash
   python src/generate_data.py
   ```

2. Generate visualizations:
   ```bash
   python src/generate_visualizations.py
   ```

3. Train models:
   ```bash
   python train_kpt_standalone.py
   ```

4. Compile LaTeX report:
   ```bash
   cd latex
   pdflatex main.tex
   pdflatex main.tex
   pdflatex main.tex
   ```

## Key Files

- **`train_kpt_standalone.py`** - Self-contained training script (no external imports from codebase)
- **`codebase.py`** - Utility functions and neural network models
- **`latex/main.pdf`** - Full 31-page technical report
- **`models/feature_importance.csv`** - Feature rankings from trained model
- **`models/model_improvements.csvMain model training script (self-contained, includes 5 new features

## Scalability

Solution scales across merchant tiers:
- **Tier 1 (Small):** App-only enhancements (₹0 hardware)
- **Tier 2 (Medium):** Lite IoT sensors (₹8K hardware)
- **Tier 3 (Large):** Full IoT suite (₹15K hardware)
- **Tier 4 (Enterprise):** Custom solutions (₹25K+ hardware)

## Documentation

Complete technical documentation available in:
- **PDF Report:** `latex/main.pdf` (31 pages)
- **Analysis Notebook:** Available via Google Drive link below

## Links

- **GitHub Repository:** https://github.com/atrishmanm/Zomathon
- **Datasets:** https://drive.google.com/drive/folders/1TptNTgQ0TjoBT6p-4mqq8uRUiuxfotVA
- **Jupyter Notebook:** https://drive.google.com/file/d/1FQOoFvgccc0uOL7AzOMgMYq6fmnWkGRS

## License

MIT License - see repository for details

---

**Prepared for Zomathon 2026 | Problem Statement 1**
