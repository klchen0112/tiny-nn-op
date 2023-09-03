#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#include <cmath>
#include <inttypes.h>
#define WarpSize 32

bool CheckResult(int8_t *out, int8_t* groudtruth, int N){
    for (int i = 0; i < N; i++){
      if(i == 0){
        printf("1st comparsion: %d and %d \n", out[i], groudtruth[i] );
      }
      if (out[i] != groudtruth[i]) {
          return false;
      }
    }
    return true;
}

int8_t* quantSymCPU(float* input, int8_t* result, int rows, int cols){
  for (int j = 0; j < rows; j++)
  {
    float MAX = 0;
    for(int i = 0; i < cols; i++)
    {
      MAX = max(abs(input[j * cols + i]), MAX);
    }
    float scalew = 127 / MAX;
    for(int i = 0; i < cols; i++)
    {
      result[j * cols + i] = round(scalew * input[j * cols + i]);
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



template<int colsPerThread>
__global__ void WarpQuant(float* d_in,int8_t* d_out,int rows,int cols) {
    const int global_warp_id = blockIdx.y * blockDim.y + threadIdx.y;

    // cal warp level max value
    float maxV = d_in[global_warp_id];
    for (int i = global_warp_id;i < global_warp_id + colsPerThread;i++) {
        maxV = max(maxV,d_in[i]);
    }
    maxV = WarpMax<colsPerThread>(maxV);
    __syncwarp();

    float scalew = 127 / maxV;
    for (int i = global_warp_id;i < global_warp_id + colsPerThread;i++) {
        d_out[i] = round(scalew * d_in[i]);
    }
}

int main(){
    int8_t a = 0;
    float milliseconds = 0;

    const int rows = 1000;
    const int cols = 1024;
    const int N = rows * cols;
    float *src = (float *)malloc(N * sizeof(float));
    float *d_src;
    cudaMalloc((void **)&d_src, N * sizeof(float));

    //int gridSize = ;//2d block, blockx=32,blocky=num warps in a block,griddimy=block nums
    //int blockSize = 256;
    int8_t *dst = (int8_t*)malloc(N * sizeof(int8_t));
    int8_t *d_dst;
    cudaMalloc((void **)&d_dst, N * sizeof(int8_t));
    int8_t *groudtruth = (int8_t *)malloc(N * sizeof(int8_t));

    for(int i = 0; i < N; i++){
        src[i] = i % cols;
    }

    groudtruth = quantSymCPU(src, dst, rows, cols);

    cudaMemcpy(d_src, src, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(1, 1000);//y轴1000个block,
    dim3 Block(32, 32);//x轴32个threads组成一个warp访问一列,

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    WarpQuant<32><<<Grid, Block>>>(d_src, d_dst, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(dst, d_dst, N * sizeof(int8_t), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(dst, groudtruth, N);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i=0;i<10;i++){
            printf("%d " ,(int)dst[i]);
        }
        printf("\n");
    }
    printf("Symeetric latency = %f ms\n", milliseconds);

    cudaFree(d_src);
    cudaFree(d_dst);
    free(src);
    free(dst);
    free(groudtruth);
    return 0;
}
