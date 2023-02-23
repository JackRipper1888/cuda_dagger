#include <stdio.h>


__global__ void kernel1( )
{
    int th_index = blockIdx.x*blockDim.x + threadIdx.x;
    printf("-------> kernel1 thread number: %d \n", th_index);
}

__global__ void sub_kernel( )
{
    int th_index = blockIdx.x*blockDim.x + threadIdx.x;
    printf("-------> sub_kernel thread number: %d \n", th_index);
}
 
__global__ void  kernel2( )
{
    int th_index = blockIdx.x*blockDim.x + threadIdx.x;
    printf("-------> kernel2 thread number: %d \n", th_index);
    sub_kernel<<<2,2>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
           printf("HashToPolynomia Blake2s256 CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

__device__  __host__ 
int add(int a, int b, int c = 1){
     #ifdef __CUDA_ARCH__
        return a + b;
     #else
           return a + b + c;
     #endif
}
__global__ void  kernel3( )
{
    int th_index = blockIdx.x*blockDim.x + threadIdx.x;
    printf("-------> kernel3 thread number: %d + 1 = %d \n", th_index, add(th_index, 1));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
           printf("HashToPolynomia Blake2s256 CUDA Error: %s\n", cudaGetErrorString(err));
    }
}
int main(void )
{
    kernel1<<<1,1>>>();
    kernel2<<<2,2>>>();
    kernel3<<<1,2>>>();
    printf("host add = %d", add(1,1));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
           printf("HashToPolynomia Blake2s256 CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceReset();
    return 0;
}