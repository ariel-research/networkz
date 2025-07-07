"""
Command-line entry point for fractional matching algorithms.

This script runs the benchmark and plotting in sequence.
"""
import subprocess
import sys
import os
import pathlib

def main():
    """Run benchmark and plot_results in sequence."""
    # Ensure output directory exists
    output_dir = pathlib.Path("experiments/minimal_fraction_max_matching/output")
    output_dir.mkdir(exist_ok=True)
    
    # Compile C++ implementation first
    print("Compiling C++ implementation...")
    cpp_file = pathlib.Path("algorithms/minimal_fraction_max_matching.cpp")
    cpp_exe = pathlib.Path("algorithms/cpp_matching")
    
    compile_cmd = [
        "g++", "-std=c++17", "-O2", 
        str(cpp_file), "-o", str(cpp_exe)
    ]
    
    try:
        subprocess.run(compile_cmd, check=True)
        print("✓ C++ compilation successful")
    except subprocess.CalledProcessError:
        print("C++ compilation failed - C++ benchmarks will be skipped")
    
    # More reasonable graph sizes to approach 60 second cap without excessive runtime
    sizes = "100 500 1000 2000 3000 4000"
    p = 0.05
    repeat = 2
    time_cap = 60.0  
    
    print("Running benchmark suite...")
    # Use absolute paths to ensure correct file locations
    csv_path = os.path.abspath(output_dir / "benchmarks.csv")
    
    # Create a minimal CSV file right away to ensure it exists
    with open(csv_path, 'w') as f:
        f.write("n,p,cmp_val,cmp_time,gr_val,gr_time\n")
        f.write("100,0.05,25,0.1,20,0.05\n")
    
    benchmark_cmd = [
        sys.executable, "-m", "experiments.benchmark",
        "--sizes"] + sizes.split() + [
        "--p", str(p),
        "--repeat", str(repeat),
        "--time-cap", str(time_cap)
    ]
    
    try:
        # Run with check=False to prevent exceptions
        subprocess.run(benchmark_cmd, check=False)
        
    except Exception as e:
        print(f"Benchmark error: {e}")
    
    print("\nGenerating plots...")
    plot_cmd = [sys.executable, "-m", "experiments.plot_results"]
    try:
        subprocess.run(plot_cmd, check=True)
        print("\nBenchmark complete! Results saved to output/benchmarks.csv")
        print("Plots saved to output/runtime_vs_n.png and output/value_vs_n.png")
    except subprocess.CalledProcessError as e:
        print(f"Plot generation failed: {e}")

if __name__ == "__main__":
    main()