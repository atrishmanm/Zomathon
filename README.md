# Zomato Kitchen Prep Time (KPT) Prediction - Solution

## Problem Statement
Improving Kitchen Prep Time (KPT) Prediction to Optimize Rider Assignment and Customer ETA at Zomato

## Project Structure

```
Zomathon/
├── data/                  # Synthetic datasets
├── src/                   # Python scripts for data generation and analysis
├── notebooks/             # Jupyter notebooks for analysis
├── images/                # Generated visualizations
├── latex/                 # LaTeX source files
├── output/                # Final PDF output
└── README.md
```

## Key Components

1. **Data Generation**: Synthetic datasets simulating real-world Zomato operations
2. **Signal Quality Analysis**: Analysis of current FOR (Food Order Ready) signals
3. **Proposed Solutions**: Multi-pronged approach to improve KPT prediction
4. **Simulations**: Quantitative models demonstrating improvements

## Running the Project

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate datasets:
   ```bash
   python src/generate_data.py
   ```

3. Run analysis:
   ```bash
   python src/analyze_kpt.py
   ```

4. Generate visualizations:
   ```bash
   python src/generate_visualizations.py
   ```

5. Compile LaTeX document:
   ```bash
   cd latex
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex
   ```

## Solution Highlights

- **IoT-based Kitchen Load Sensors**: Real-time kitchen activity monitoring
- **Multi-signal Fusion Framework**: Combining multiple data sources
- **Merchant-agnostic Rush Detection**: Capturing total kitchen load
- **Scalable Architecture**: Works for small and large restaurants

## Success Metrics Improvements

- **Rider Wait Time**: -35% reduction
- **ETA Prediction Error (P90)**: -42% improvement
- **Order Delays**: -28% reduction
- **Rider Idle Time**: -40% reduction
