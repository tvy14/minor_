import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Define the CUDA kernel function
cuda_code = """
__global__ void parallel_threading(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Perform computation using the GPU
        output[idx] = input[idx] * input[idx];
    }
}
"""

# Compile the CUDA kernel function
mod = SourceModule(cuda_code)

# Get the compiled CUDA kernel function
parallel_threading = mod.get_function("parallel_threading")

# Define the input data
input_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
output_data = np.zeros_like(input_data)

# Allocate memory on the GPU
input_gpu = cuda.mem_alloc(input_data.nbytes)
output_gpu = cuda.mem_alloc(output_data.nbytes)

# Transfer the input data to the GPU
cuda.memcpy_htod(input_gpu, input_data)

# Define the block and grid dimensions
block_size = 128
grid_size = (input_data.size + block_size - 1) // block_size

# Launch the CUDA kernel on the GPU
parallel_threading(input_gpu, output_gpu, np.int32(input_data.size),
                   block=(block_size, 1, 1), grid=(grid_size, 1))

# Transfer the output data from the GPU
cuda.memcpy_dtoh(output_data, output_gpu)

# Print the result
print("Input:", input_data)
print("Output:", output_data)
