#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#include <cmath>

#define WarpSize 32

bool CheckResult(float *out, float* groudtruth, int N){
    for (int i = 0; i < N; i++){
      if(i == 0){
        printf("1st comparsion: %f and %f \n" , out[i], groudtruth[i] );
      }
      if (out[i] != groudtruth[i]) {
          return false;
      }
    }
    return true;
}

float* softmaxCPU(float* input, float* result, int rows, int cols){
  for (int j = 0; j < rows; j++)
  {
    float total = 0;
    float MAX = 0;
    for(int i = 0; i < cols; i++)
    {
      MAX = max(input[j * cols + i], MAX);
    }
    for(int i = 0; i < cols; i++)
    {
      total += exp(input[j * cols + i] - MAX);
    }
    for(int i = 0; i < cols; i++)
    {
      result[j * cols + i] = exp(input[j * cols + i] - MAX) / total;
    }
  }

  return result;
}


template<int blockSize>
__device__ float WarpMax(float maxV) {
    if (blockSize >= 32 ) maxV = max(maxV,__shfl_down_sync(0xffffffff,maxV,16));
    if (blockSize >= 16 ) maxV = max(maxV,__shfl_down_sync(0xffffffff,maxV,8));
    if (blockSize >= 8 ) maxV = max(maxV,__shfl_down_sync(0xffffffff,maxV,4));
    if (blockSize >= 4 ) maxV = max(maxV,__shfl_down_sync(0xffffffff,maxV,2));
    if (blockSize >= 2 ) maxV = max(maxV,__shfl_down_sync(0xffffffff,maxV,1));
    return maxV;
}

template<int blockSize>
__device__ float WarpSum(float sum) {
    //__shfl_down_sync：前面的thread向后面的thread要数据
    //__shfl_up_sync: 后面的thread向前面的thread要数据
    //返回前面的thread向后面的thread要的数据，比如__shfl_down_sync(0xffffffff, sum, 16)那就是返回16号线程，17号线程的数据
    //warp内的数据交换不会出现warp在shared memory上交换数据时的不一致现象，无需syncwarp
    if (blockSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}


template<int colsPerThread>
__global__ void WarpSoftmax(float* d_in,float* d_out,int rows,int cols) {
    const int global_warp_id = blockIdx.y * blockDim.y + threadIdx.y;

    // cal warp level max value
    float maxV = d_in[global_warp_id];
    for (int i = global_warp_id;i < global_warp_id + colsPerThread;i++) {
        maxV = max(maxV,d_in[i]);
    }
    maxV = WarpMax<colsPerThread>(maxV);
    __syncwarp();

    float sum = 0;

    for (int i = global_warp_id;i < global_warp_id + colsPerThread;i++) {
        d_out[i] = d_in[i] / maxV;
        sum = sum + d_out[i];
    }
    sum = WarpSum<32>(sum);
    __syncwarp();
    for (int i = global_warp_id;i < global_warp_id + colsPerThread;i++) {
        d_out[i] = d_out[i] / sum ;
    }
}

int main(){
    float milliseconds = 0;
    const int N = 1000 * 1024;
    const int rows = 1000;
    const int cols = 1024;
    float *src = (float *)malloc(N * sizeof(float));
    float *d_src;
    cudaMalloc((void **)&d_src, N * sizeof(float));

    //int gridSize = ;//2d block, blockx=32,blocky=num warps in a block,griddimy=block nums
    //int blockSize = 256;
    float *dst = (float*)malloc(N * sizeof(float));
    float *d_dst;
    cudaMalloc((void **)&d_dst, N * sizeof(float));
    float *groudtruth = (float *)malloc(N * sizeof(float));

    for(int i = 0; i < N; i++){
        src[i] = 1;
    }

    groudtruth = softmaxCPU(src, dst, rows, cols);

    cudaMemcpy(d_src, src, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(1, 1000);//y轴1000个block,
    dim3 Block(32, 32);//x轴32个threads组成一个warp访问一列,

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    WarpSoftmax<32><<<Grid, Block>>>(d_src, d_dst, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(dst, groudtruth, N);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i=0;i<10;i++){
            printf("%lf ",dst[i]);
        }
        printf("\n");
    }
    printf("WarpSoftmax latency = %f ms\n", milliseconds);

    cudaFree(d_src);
    cudaFree(d_dst);
    free(src);
    free(dst);
    free(groudtruth);
}
