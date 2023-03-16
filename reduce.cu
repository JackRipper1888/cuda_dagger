#include <stdio.h>

__global__ void SerialKernel(const float* input, float* output, size_t n) {
  float sum = 0.0f;  
  for (size_t i = 0; i < n; ++i) {  
    sum += input[i];  
  }  
  *output = sum;  
}

void ReduceBySerial(const float* input, float* output, size_t n) {
  SerialKernel<<<1, 1>>>(input, output, n);
}

__global__ void AtomicKernel(const float* input, float* output, size_t n) {
  int32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  atomicAdd(output,input[gtid]);
}

void ReduceByAtomic(const float* input, float* output, size_t n) {
    AtomicKernel<<<n/1024, 1024>>>(input, output, n);
}

__global__ void TwoPassSimpleKernel(const float* input, float* part_sum,
                                    size_t n) {
  // n is divided to gridDim.x part
  // this block process input[blk_begin:blk_end]
  // store result to part_sum[blockIdx.x]
  size_t blk_begin = n / gridDim.x * blockIdx.x;
  size_t blk_end = n / gridDim.x * (blockIdx.x + 1);
  // after follow step, this block process input[0:n], store result to part_sum
  n = blk_end - blk_begin;
  input += blk_begin;
  part_sum += blockIdx.x;
  // n is divided to blockDim.x part
  // this thread process input[thr_begin:thr_end]
  size_t thr_begin = n / blockDim.x * threadIdx.x;
  size_t thr_end = n / blockDim.x * (threadIdx.x + 1);
  float thr_sum = 0.0f;
  for (size_t i = thr_begin; i < thr_end; ++i) {
    thr_sum += input[i];
  }
  // store thr_sum to shared memory
  extern __shared__ float shm[];
  shm[threadIdx.x] = thr_sum;
  __syncthreads();
  // reduce shm to part_sum
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (size_t i = 0; i < blockDim.x; ++i) {
      sum += shm[i];
    }
    *part_sum = sum;
  }
}

void ReduceByTwoPass(const float* input, float* output, size_t n) {
  const int32_t thread_num_per_block = 1024;  // tuned
  const int32_t block_num = 1024;             // tuned
  float *part = NULL;
  cudaMalloc((void**)&part,(1024*sizeof(float)));
  // the first pass reduce input[0:n] to part[0:block_num]
  // part_sum[i] stands for the result of i-th block
  size_t shm_size = thread_num_per_block * sizeof(float);  // float per thread
  TwoPassSimpleKernel<<<block_num, thread_num_per_block, shm_size>>>(input,part, n);
  // the second pass reduce part[0:block_num] to output
  TwoPassSimpleKernel<<<1, thread_num_per_block, shm_size>>>(part, output, block_num);
}



// warp divergence
__global__ void TwoPassInterleavedKernel(const float* input, float* part_sum,
                                         size_t n) {
  int32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  int32_t total_thread_num = gridDim.x * blockDim.x;
  // reduce
  //   input[gtid + total_thread_num * 0]
  //   input[gtid + total_thread_num * 1]
  //   input[gtid + total_thread_num * 2]
  //   input[gtid + total_thread_num * ...]
  float sum = 0.0f;
  for (int32_t i = gtid; i < n; i += total_thread_num) {
    sum += input[i];
  }
  // store sum to shared memory
  extern __shared__ float shm[];
  shm[threadIdx.x] = sum;
  __syncthreads();
  // reduce shm to part_sum
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (size_t i = 0; i < blockDim.x; ++i) {
      sum += shm[i];
    }
    part_sum[blockIdx.x] = sum;
  }
}


void ReduceByTwoPassInterleaved(const float* input, float* output, size_t n) {
  const int32_t thread_num_per_block = 1024;  // tuned
  const int32_t block_num = 1024;             // tuned
  float *part = NULL;
  cudaMalloc((void**)&part,(1024*sizeof(float)));
  // the first pass reduce input[0:n] to part[0:block_num]
  // part_sum[i] stands for the result of i-th block
  size_t shm_size = thread_num_per_block * sizeof(float);  // float per thread
  TwoPassInterleavedKernel<<<block_num, thread_num_per_block, shm_size>>>(input,part, n);
  // the second pass reduce part[0:block_num] to output
  TwoPassInterleavedKernel<<<1, thread_num_per_block, shm_size>>>(part, output, block_num);
}

__global__ void baselineKernel(const float* input, float* part_sum,
                                         size_t n) {
  int32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  int32_t total_thread_num = gridDim.x * blockDim.x;
  float sum = 0.0f;
  for (int32_t i = gtid; i < n; i += total_thread_num) {
    sum += input[i];
  }
  // store sum to shared memory
  extern __shared__ float shm[];
  shm[threadIdx.x] = sum;
  __syncthreads();
  // reduce shm to part_sum
  for(unsigned int s=1; s<blockDim.x; s*=2){
    if(threadIdx.x%(2*s) == 0){
        shm[threadIdx.x]+=shm[threadIdx.x+s];
    }
    __syncthreads();
  }
    
 if (threadIdx.x == 0) part_sum[blockIdx.x] = shm[0];
}

void ReduceByBaseline(const float* input, float* output, size_t n) {
  const int32_t thread_num_per_block = 1024;  // tuned
  const int32_t block_num = 1024;             // tuned
  float *part = NULL;
  cudaMalloc((void**)&part,(1024*sizeof(float)));
  // the first pass reduce input[0:n] to part[0:block_num]
  // part_sum[i] stands for the result of i-th block
  size_t shm_size = thread_num_per_block * sizeof(float);  // float per thread
  baselineKernel<<<block_num, thread_num_per_block, shm_size>>>(input,part, n);
  // the second pass reduce part[0:block_num] to output
  baselineKernel<<<1, thread_num_per_block, shm_size>>>(part, output, block_num);
}

__global__ void warpBranchKernel(const float* input, float* part_sum,
                                         size_t n) {
  int32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  int32_t total_thread_num = gridDim.x * blockDim.x;
  float sum = 0.0f;
  for (int32_t i = gtid; i < n; i += total_thread_num) {
    sum += input[i];
  }
  // store sum to shared memory
  extern __shared__ float shm[];
  shm[threadIdx.x] = sum;
  __syncthreads();
  // reduce shm to part_sum
  for(unsigned int s=1; s<blockDim.x; s*=2){
    int index = 2*s*threadIdx.x;
    if(index < blockDim.x){
        shm[index]+=shm[index+s];
    }
    __syncthreads();
}
    
 if (threadIdx.x == 0) part_sum[blockIdx.x] = shm[0];
}

void ReduceByWarpBranch(const float* input, float* output, size_t n) {
  const int32_t thread_num_per_block = 1024;  // tuned
  const int32_t block_num = 1024;             // tuned
  float *part = NULL;
  cudaMalloc((void**)&part,(1024*sizeof(float)));
  // the first pass reduce input[0:n] to part[0:block_num]
  // part_sum[i] stands for the result of i-th block
  size_t shm_size = thread_num_per_block * sizeof(float);  // float per thread
  warpBranchKernel<<<block_num, thread_num_per_block, shm_size>>>(input,part, n);
  // the second pass reduce part[0:block_num] to output
  warpBranchKernel<<<1, thread_num_per_block, shm_size>>>(part, output, block_num);
}

__global__ void TwoPassSharedOptimizedKernel(const float* input,
                                             float* part_sum, size_t n) {
  int32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  int32_t total_thread_num = gridDim.x * blockDim.x;
  // reduce
  //   input[gtid + total_thread_num * 0]
  //   input[gtid + total_thread_num * 1]
  //   input[gtid + total_thread_num * 2]
  //   input[gtid + total_thread_num * ...]
  float sum = 0.0f;
  for (int32_t i = gtid; i < n; i += total_thread_num) {
    sum += input[i];
  }
  // store sum to shared memory
  extern __shared__ float shm[];
  shm[threadIdx.x] = sum;
  __syncthreads();
  // reduce shm to part_sum
  for (int32_t active_thread_num = blockDim.x / 2; active_thread_num >= 1;
       active_thread_num /= 2) {
    if (threadIdx.x < active_thread_num) {
      shm[threadIdx.x] += shm[threadIdx.x + active_thread_num];
    }
    __syncthreads();
  }
  //   Shared Memory 有 4 字节模式和 8 字节模式：
  // 4 字节模式：其中属于 Bank 0 的地址有 [0, 4), [128, 132), [256, 260)...，而属于 Bank 1 的地址有 [4, 8), [132, 136), [260, 264) ...，依次类推每个 bank 的地址。
  // 8 字节模式：其中属于 Bank 0 的地址有 [0, 8), [256, 264), [512, 520)...，而属于 Bank 1 的地址有 [8, 16), [264, 272), [520, 528) ...，依次类推每个 bank 的地址。
  //   Bank  |      1      |      2      |      3      |...
  // Address |  0  1  2  3 |  4  5  6  7 |  8  9 10 11 |...
  // Address | 64 65 66 67 | 68 69 70 71 | 72 73 74 75 |...
  //    
  // 0 + 4 , 1 + 5, 2 + 6 3 + 7
  // 0 + 2 , 1 + 3   
  if (threadIdx.x == 0) {
    part_sum[blockIdx.x] = shm[0];
  }
}

void ReduceByTwoPassSharedOptimized(const float* input, float* output,size_t n) {
  const int32_t thread_num_per_block = 1024;  // tuned
  const int32_t block_num = 1024;             // tuned
  float *part = NULL;
  cudaMalloc((void**)&part,(1024*sizeof(float)));
  // the first pass reduce input[0:n] to part[0:block_num]
  // part_sum[i] stands for the result of i-th block
  size_t shm_size = thread_num_per_block * sizeof(float);  // float per thread
  TwoPassSharedOptimizedKernel<<<block_num, thread_num_per_block, shm_size>>>(input,part, n);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("ReduceByTwoPassSharedOptimized 1 CUDA Error: %s\n", cudaGetErrorString(err));
  }
  // the second pass reduce part[0:block_num] to output
  TwoPassSharedOptimizedKernel<<<1, thread_num_per_block, shm_size>>>(part, output, block_num);
}


int main() 
{
    float *data = NULL;
    cudaMallocHost((void**)&data, (4*1024*1024*sizeof(float)));
    for (int i=0; i<4*1024*1024; i++) {
        data[i] = (float)i;
    }

    float *input = NULL;
    float *output = NULL;
    cudaMalloc((void**)&input,(4*1024*1024*sizeof(float)));
    cudaMalloc((void**)&output,sizeof(float));

    // HtoD;
    cudaMemcpy(input, data, 4*1024*1024*sizeof(float), cudaMemcpyHostToDevice);

    ReduceBySerial(input, output, 4*1024*1024);

    ReduceByAtomic(input, output, 4*1024*1024);

    ReduceByTwoPassInterleaved(input, output, 4*1024*1024);

    ReduceByBaseline(input, output, 4*1024*1024);

    ReduceByWarpBranch(input, output, 4*1024*1024);

    ReduceByTwoPassSharedOptimized(input, output, 4*1024*1024);

    cudaDeviceSynchronize();
    return 0;
}