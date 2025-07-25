import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define matrix size for a compute-bound operation
matrix_size = 30000
a = torch.randn(matrix_size, matrix_size, device=device)
b = torch.randn(matrix_size, matrix_size, device=device)

# Warm-up runs
for _ in range(10):
    _ = torch.matmul(a, b)

# Measure actual time
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
c = torch.matmul(a, b)
end_event.record()

torch.cuda.synchronize()
elapsed_time_ms = start_event.elapsed_time(end_event) # Time in milliseconds

# Calculate FLOPs for matrix multiplication (2 * N^3 for N x N matrices)
flops = 2 * (matrix_size ** 3)
actual_tflops = (flops / elapsed_time_ms) * 1e-9 # Convert to TFLOPS

print(f"Actual TFLOPS: {actual_tflops:.2f}")
