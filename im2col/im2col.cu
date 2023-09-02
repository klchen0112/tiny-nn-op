#include <cmath>
#include <cstddef>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define DEBUG
#ifdef DEBUG
#define checkCudaErrors(func)                                                  \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    if (e != cudaSuccess)                                                      \
      printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));   \
  }
#else
#define checkCudaErrors(func) (func)
#endif

template <typename T> struct Tensor {
  T *data;
  int *shape;
  int *stride;
  int ndim;
};

template<typename T>
__global__ void im2col(Tensor<T>* out,Tensor<T>* src,int kh,int kw,int sh,int sw,int dh,int dw,int ph,int pw) {
    const int C = src->shape[0],
    H= src->shape[1],
    W = src->shape[2];
    const int num_kernels = C * out->shape[0] * out->shape[1];
    const int width_col = (H + 2 * ph - (dh * (kh - 1)) - 1) / sh + 1;
    const int height_col = (W + 2 * pw - (dw * (kw - 1)) - 1) / sw + 1;


    for (int index = blockDim.x * blockIdx.x + threadIdx.x;index < num_kernels;index += blockDim.x * gridDim.x){
        int w_out = index % width_col;
        int idx = index / width_col;
        int h_out = idx % height_col;
        int channel_in = idx / height_col;
        int channel_out = channel_in * kh * kw;

        int h_in = h_out * sh - ph;
        int w_in = w_out * sw - pw;

        T* col = out->data + (channel_out * height_col + h_out) * width_col + w_out;
        const T* im = src->data + (channel_in * H + h_in) * W + w_in;
        for (int i = 0;i<kw;i++) {
            int h = h_in + i * dh;
            for (int j = 0;j < kw;j++) {
                int w = w_in + j * dw;
                if (h >= 0 && w >= 0 && h < H && w < W) {
                    *col = im[i * dh * W + j * dw];
                } else {
                    *col = 0;
                }
                col = col + height_col * width_col;
            }
        }
    }
}

int main(int argc, char **argv) {

  if (argc != 12) {
    printf("usage: ./main  [C] [H] [W] [kh] [kw] [sh] [sw] [dh] [dw] [ph] [pw]\n");
    exit(0);
  }
  const int nDim = 3;
  const int C = atoi(argv[1]);
  const int H = atoi(argv[2]);
  const int W = atoi(argv[3]);

  const int kh = atoi(argv[4]);
  const int kw = atoi(argv[5]);

  const int sh = atoi(argv[6]);
  const int sw = atoi(argv[7]);

  const int dh = atoi(argv[8]);
  const int dw = atoi(argv[9]);

  const int ph = atoi(argv[10]);
  const int pw = atoi(argv[11]);


  Tensor<int> *cTensor = (Tensor<int> *)malloc(sizeof(Tensor<int>));
  cTensor->ndim = nDim;
  cTensor->shape = (int *)malloc(sizeof(int) * cTensor->ndim);
  cTensor->shape[0] = C;
  cTensor->shape[1] = H;
  cTensor->shape[2] = W;

  cTensor->stride = (int *)malloc(sizeof(int) * cTensor->ndim);

  cTensor->stride[2] = 1;
  cTensor->stride[1] = cTensor->stride[2] * cTensor->shape[2];
  cTensor->stride[0] = cTensor->stride[1] * cTensor->shape[1];



  const int sumElements = cTensor->stride[0] * cTensor->shape[0];
  cTensor->data = (int *)malloc(sumElements * sizeof(int));
  for (int ele = 0; ele < sumElements; ele++) {
    cTensor->data[ele] = ele;
  }
  printf("src Tensor sumElements %d shape (%d,%d,%d) stride (%d,%d,%d)\n",
         sumElements,
          cTensor->shape[0], cTensor->shape[1], cTensor->shape[2],
          cTensor->stride[0], cTensor->stride[1],cTensor->stride[2]);

  Tensor<int> *cgTensor = (Tensor<int> *)malloc(sizeof(Tensor<int>));

  cgTensor->ndim = nDim;

  checkCudaErrors(cudaMalloc(&cgTensor->shape, sizeof(int) * cTensor->ndim));
  checkCudaErrors(cudaMemcpy(cgTensor->shape, cTensor->shape,
                             sizeof(int) *nDim, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&cgTensor->stride, sizeof(int) * cTensor->ndim));
  checkCudaErrors(cudaMemcpy(cgTensor->stride, cTensor->stride,
                             sizeof(int) *nDim, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&cgTensor->data, sizeof(int) * sumElements));
  checkCudaErrors(cudaMemcpy(cgTensor->data, cTensor->data,
                             sizeof(int) * sumElements,
                             cudaMemcpyHostToDevice));

  Tensor<int> *gTensor;

  checkCudaErrors(cudaMalloc(&gTensor, sizeof(Tensor<int>)));
  checkCudaErrors(cudaMemcpy(gTensor, cgTensor, sizeof(Tensor<int>),
                             cudaMemcpyHostToDevice));

  printf("gen gTensor src ok\n");
  const int Hout = (H + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
  const int Wout = (H + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

  Tensor<int>* cOut = (Tensor<int> *)malloc(sizeof(Tensor<int>));

  cOut->ndim = 2;

  cOut->shape = (int *)malloc(sizeof(int) * cOut->ndim);
  cOut->shape[0] = Hout * Wout;
  cOut->shape[1] = C * kh * kw;

  cOut->stride = (int *)malloc(sizeof(int) * cOut->ndim);
  cOut->stride[1] = 1;
  cOut->stride[0] = cTensor->stride[3] * cTensor->shape[3];

  const int sumOutElements = cOut->stride[0] * cOut->shape[0];
  cOut->data = (int *)malloc(sumOutElements * sizeof(int));

  printf("dst Tensor sumElements %d shape (%d,%d) stride (%d,%d)\n",
         sumOutElements, cOut->shape[0], cOut->shape[1],
          cOut->stride[0], cOut->stride[1]
        );

  Tensor<int> *cgOut = (Tensor<int> *)malloc(sizeof(Tensor<int>));

  cgOut->ndim = cOut->ndim;

  checkCudaErrors(cudaMalloc(&cgOut->shape, sizeof(int) * cgOut->ndim));
  checkCudaErrors(cudaMemcpy(cgOut->shape, cgOut->shape,
                             sizeof(int) *nDim, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&cgOut->stride, sizeof(int) * cgOut->ndim));
  checkCudaErrors(cudaMemcpy(cgOut->stride, cgOut->stride,
                             sizeof(int) *nDim, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&cgOut->data, sizeof(int) * sumOutElements));
  checkCudaErrors(cudaMemcpy(cgOut->data, cTensor->data,
                             sizeof(int) * sumOutElements,
                             cudaMemcpyHostToDevice));

  Tensor<int> *gOut;

  checkCudaErrors(cudaMalloc(&gOut, sizeof(Tensor<int>)));
  checkCudaErrors(cudaMemcpy(gOut, cgOut, sizeof(Tensor<int>),
                             cudaMemcpyHostToDevice));

  printf("gen gTensor src ok\n");



  float milliseconds = 0;
  cudaEvent_t start, stop;
  (cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));
  for (int i = 0; i < 100; i++) {
    dim3 grid((C*Hout * Wout + 1024 - 1) / 1024);
    dim3 block(1024);
    im2col<int><<<grid,block>>>(gOut, gTensor,kh,kw,sh,sw,dh,dw,ph,pw);
  }
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("record and run complete %f ms\n", milliseconds);
  printf("im2col ok\n");
  checkCudaErrors(cudaMemcpy(cOut->data, cgOut->data, sizeof(int) * sumOutElements,
                             cudaMemcpyDeviceToHost));

  printf("cal ok\n");

  return 0;
}
