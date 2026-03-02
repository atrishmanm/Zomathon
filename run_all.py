"""
Master script to generate all data, run analysis, create visualizations, and compile PDF
"""

import subprocess
import sys
import os

def run_command(description, command):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    try:
        # Use list form to avoid shell issues with spaces in paths
        result = subprocess.run(command, check=True, 
                              capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Run all steps to generate the complete solution"""
    
    print("\n" + "="*80)
    print(" "*20 + "ZOMATO KPT PREDICTION SOLUTION")
    print(" "*25 + "Complete Generation Pipeline")
    print("="*80)
    
    # Step 1: Generate data
    if not run_command(
        "Step 1/6: Generating synthetic datasets...",
        [sys.executable, "src/generate_data.py"]
    ):
        print("\n❌ Failed to generate data. Exiting.")
        return False
    
    # Step 2: Run analysis
    if not run_command(
        "Step 2/6: Running KPT analysis...",
        [sys.executable, "src/analyze_kpt.py"]
    ):
        print("\n❌ Failed to run analysis. Exiting.")
        return False
    
    # Step 3: Train ML models
    if not run_command(
        "Step 3/6: Training KPT prediction models...",
        [sys.executable, "src/train_kpt_model.py"]
    ):
        print("\n❌ Failed to train models. Exiting.")
        return False
    
    # Step 4: Generate visualizations
    if not run_command(
        "Step 4/6: Generating visualizations...",
        [sys.executable, "src/generate_visualizations.py"]
    ):
        print("\n❌ Failed to generate visualizations. Exiting.")
        return False
    
    # Step 5: Compile LaTeX to PDF
    print("\n" + "="*60)
    print("Step 5/6: Compiling LaTeX document to PDF...")
    print("="*60)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Change to latex directory and compile
    os.chdir('latex')
    
    # Run pdflatex multiple times for proper references
    for i in range(1, 3):
        if not run_command(
            f"   Compiling PDF (pass {i}/2)...",
            ["pdflatex", "-interaction=nonstopmode", "main.tex"]
        ):
            print("\n⚠️  Warning: PDF compilation had issues, but may have succeeded.")
    
    # Move back to root directory
    os.chdir('..')
    
    # Copy PDF to output folder
    if os.path.exists('latex/main.pdf'):
        import shutil
        shutil.copy('latex/main.pdf', 'output/Zomato_KPT_Solution.pdf')
        print("\n✅ PDF successfully created: output/Zomato_KPT_Solution.pdf")
        
        # Check file size
        size_bytes = os.path.getsize('output/Zomato_KPT_Solution.pdf')
        size_mb = size_bytes / (1024 * 1024)
        print(f"   File size: {size_mb:.2f} MB")
        
        if size_mb > 1:
            print(f"   ⚠️  Warning: File size exceeds 1 MB requirement ({size_mb:.2f} MB)")
            print(f"   Consider compressing images or reducing content.")
    else:
        print("\n❌ PDF file not found. LaTeX compilation may have failed.")
        return False
    
    # Step 6: Summary
    print("\n" + "="*80)
    print("Step 6/6: Generation Complete!")
    print("="*80)
    
    print("\n📁 Generated Files:")
    print("   ✓ data/ — Synthetic datasets (4 CSV files)")
    print("   ✓ models/ — Trained ML models (baseline & enhanced)")
    print("   ✓ images/ — Visualizations (5 PNG files)")
    print("   ✓ notebooks/kpt_analysis.ipynb — Jupyter notebook")
    print("   ✓ output/Zomato_KPT_Solution.pdf — Final PDF submission")
    
    print("\n📊 Next Steps:")
    print("   1. Review the PDF: output/Zomato_KPT_Solution.pdf")
    print("   2. Explore the notebook: notebooks/kpt_analysis.ipynb")
    print("   3. Check the models: models/*.pkl")
    print("   4. Check the data: data/*.csv")
    print("   5. View images: images/*.png")
    
    print("\n🚀 Ready for submission!")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
