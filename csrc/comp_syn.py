# Defines a comparative synapse matrix operator 
# May replace cp.dot(a,b) with cp_comp(a,b) where a,b ar 'float32' GPU operands 
# Adapted from  https://github.com/aidevnn/CuPyFirstExample
# By Radu DOGARU (radu.dogaru@upb.ro) 
# Last update Jan. 15, 2020 
#-----------------------------------------------------------------------------------------------
import cupy as cp
import numpy as np 
import math

def read_code(code_filename, params):
    with open(code_filename, 'r') as f:
        code = f.read()
    for k, v in params.items():
        code = '#define ' + k + ' ' + str(v) + '\n' + code
    return code

def benchmark(func, args, n_run):
    times = []
    for _ in range(n_run):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        func(*args)
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))  # milliseconds
    return times

# File cp_comp.cu is the CUDA kernel description in C Cuda 
# Shoud be present in the same directory 

cp_comp_file='./csrc/cp_comp.cu'


# The Python definition of the cp_comp "comparative" operator 
# The arguments MUST be 'float32' 

def cp_comp(A, B,
          dim_x=16, dim_y=16, blk_m=64, blk_n=64, blk_k=4,
          dim_xa=64, dim_ya=4, dim_xb=4, dim_yb=64):
    assert A.dtype == cp.float32
    assert B.dtype == cp.float32
    assert(dim_x * dim_y == dim_xa * dim_ya == dim_xb * dim_yb)

    m, k = A.shape
    k, n = B.shape

    # Inputs matrices need to be in Fortran order.
    # ??? why  (R. Dogaru) 

    A = cp.asfortranarray(A)
    B = cp.asfortranarray(B)

    C = cp.empty((m, n), dtype=cp.float32, order='F')

    config = {'DIM_X': dim_x, 'DIM_Y': dim_y,
              'BLK_M': blk_m, 'BLK_N': blk_n, 'BLK_K': blk_k,
              'DIM_XA': dim_xa, 'DIM_YA': dim_ya,
              'DIM_XB': dim_xb, 'DIM_YB': dim_yb,
              'THR_M': blk_m // dim_x, 'THR_N': blk_n // dim_y}
    code = read_code(cp_comp_file, params=config)
    kern = cp.RawKernel(code, 'cp_comp')

    grid = (int(math.ceil(m / blk_m)), int(math.ceil(n / blk_n)), 1)
    block = (dim_x, dim_y, 1)
    args = (m, n, k, A, B, C)
    shared_mem = blk_k * (blk_m + 1) * 4 + blk_n * (blk_k + 1) * 4
    kern(grid, block, args=args, shared_mem=shared_mem)
    return C
