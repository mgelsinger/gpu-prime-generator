# gpu-prime-generator
# GPU Prime Number Generator

This project provides a CUDA-accelerated Python script that generates an ordered list of prime numbers using an NVIDIA GPU via Numba. It streams primes to a text file and supports safe stop/restart.

## Features

- **CUDA-accelerated** prime checking using a segmented sieve approach.
- **Ordered output**: primes are written in ascending order.
- **Graceful shutdown**: press Ctrl+C to stop; the script saves the current state.
- **Resume capability**: on restart, the script continues generating from the last prime.
- **Configurable segment size**: adjust how many numbers are tested per GPU batch.

## Prerequisites

1. **Python 3.9+**  
2. **CUDA Toolkit** (Runtime + NVVM) or **conda** `cuda` package  
   - Recommended:  
     ```bash
     conda create -n primegpu python=3.11 numba cuda -c nvidia -c conda-forge
     conda activate primegpu
     pip install numpy
     ```
   - Alternatively, install the CUDA Toolkit from NVIDIA (ensure `nvvm.dll` is on your `PATH`).

3. **Numba** (0.59+ for Python 3.12; 0.61+ for Python 3.13)  
   ```bash
   pip install numba
   ```

4. **NumPy**  
   ```bash
   pip install numpy
   ```

Verify that Numba sees your GPU:
```bash
from numba import cuda
print(cuda.detect())
```

## Installation

1. Clone this repository or download the code:
   ```bash
   git clone https://github.com/yourusername/prime-num-gpu.git
   cd prime-num-gpu
   ```

2. Ensure your Python environment has the prerequisites (see above).

## Usage

Run the script with default settings:
```bash
python gpu_prime_generator.py
```
- Creates (or appends to) `primes.txt` in the current folder.
- Press **Ctrl+C** to stop. The script prints a summary, for example:
  ```
  Stopped by user. Found  784,984 primes in 12.4 s  (63 k p/s)
  ```

### Options

- `-o, --output <file>`  
  Destination file (default: `primes.txt`). If the file exists, the script resumes from the last prime.

- `-n, --segment <count>`  
  How many integers to test per GPU batch (default: `1000000`).

Example:
```bash
python gpu_prime_generator.py -o my_primes.txt -n 5000000
```

## How It Works

1. **CPU sieve (bootstrap)**  
   - Computes all primes up to √N for each segment using a standard Sieve of Eratosthenes.

2. **GPU kernel**  
   - Each CUDA thread tests one odd candidate number by checking divisibility against the pre-sieved primes.
   - Results are streamed back, filtered, and written to the output file in ascending order.

3. **Segmented loop**  
   - Numbers are processed in chunks (segments) of configurable size to manage memory usage.
   - After each segment, the script flushes new primes to disk and prints progress.

4. **Graceful shutdown / resume**  
   - On startup, the script reads the last prime in the output file (if any) and keeps a running total.
   - Pressing Ctrl+C triggers a `KeyboardInterrupt`, closing the file and printing a final summary.

## Tips & Tuning

- Increase `--segment` for fewer kernel launches (uses more GPU memory).
- Adjust `threads_per_block` in the source code (128–512 threads is typical).
- For best write performance, use an SSD or fast NVMe drive.
- Ensure only one CUDA installation is on your `PATH` to avoid conflicts.

## Troubleshooting

- **`nvvm.dll not found`**  
  Ensure the CUDA Toolkit (runtime + NVVM) is installed and its `bin` directories are on `PATH`.  
  For conda users, install the `cuda` meta-package:
  ```bash
  conda install -c nvidia -c conda-forge cuda
  ```

- **CUDA driver mismatch**  
  Make sure your GPU driver version is ≥ the CUDA Toolkit version you installed.

- **Python / Numba compatibility**  
  - Python 3.12 requires Numba 0.59+  
  - Python 3.13 requires Numba 0.61+

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
