import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

cuda_code = """
__global__ void parallel_threading(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Perform computation using the GPU
        output[idx] = input[idx] * input[idx];
    }
}
"""

mod = SourceModule(cuda_code)

parallel_threading = mod.get_function("parallel_threading")

input_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
output_data = np.zeros_like(input_data)

input_gpu = cuda.mem_alloc(input_data.nbytes)
output_gpu = cuda.mem_alloc(output_data.nbytes)

cuda.memcpy_htod(input_gpu, input_data)

block_size = 128
grid_size = (input_data.size + block_size - 1) // block_size

parallel_threading(input_gpu, output_gpu, np.int32(input_data.size),
                   block=(block_size, 1, 1), grid=(grid_size, 1))

cuda.memcpy_dtoh(output_data, output_gpu)

print("Input:", input_data)
print("Output:", output_data)
