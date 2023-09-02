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

template <typename T>
__global__ void index_select_cuda_first(Tensor<T> *out, Tensor<T> *input,
                                        Tensor<int> *indices) {
  const int nth = blockIdx.x;
  int tid = threadIdx.x;
  const int inputOffset = input->stride[0] * indices->data[nth];
  const int outOffset = input->stride[0] * blockIdx.x;
  const int nElements = input->stride[0];
  if (tid >= nElements)
    return;
  const int stride = gridDim.x;
  while (tid < nElements) {
    out->data[outOffset + tid] = input->data[inputOffset + tid];
    tid += stride;
  }
}

template <typename T>
__global__ void index_select_cuda_mid(Tensor<T> *out, Tensor<T> *input,
                                      Tensor<int> *indices, const int wchDim) {
  const int bx = blockIdx.x;
  const int tid = threadIdx.x;
  const int totalBlock = blockDim.x;
  const int totalPre =
      input->stride[0] / input->stride[wchDim - 1] * input->shape[0];
  const int inputStride = input->stride[wchDim];
  const int totalIndices = indices->shape[0];
  const int outputStride = indices->shape[0] * input->stride[wchDim];
  const int totalThread = gridDim.x;
  for (int tp = bx; tp < totalPre; tp += totalBlock) {
    int outputOffset = outputStride * tp;
    // a block threads(a warp ) copy all the elements
    for (int idx = 0; idx < totalIndices; idx++) {
      int inputOffset = tp * inputStride + idx * inputStride;
      for (int ele = tid; ele < inputStride; ele += totalThread) {
        out->data[outputOffset + idx * inputStride + ele] =
            input->data[inputOffset + ele];
      }
      // __syncwarp();
    }
  }
}

template <typename T>
__global__ void index_select_cuda_last(Tensor<T> *out, Tensor<T> *input,
                                       Tensor<int> *indices) {
  const int nTensor =
      input->stride[0] / input->shape[input->ndim - 1] * input->shape[0];
  const int inputStride = input->shape[input->ndim - 1];
  int inputOffset = inputStride * blockIdx.x;
  const int outputStride = indices->shape[0] * nTensor;
  int outputOffset = outputStride * blockIdx.x;
  for (int nth = blockIdx.x; nth < nTensor; nth += blockDim.x) {
    // a thread copy a last dim
    for (int ele = threadIdx.x; ele < indices->shape[0]; ele += gridDim.x) {
      out->data[outputOffset + ele] =
          input->data[inputOffset + indices->data[ele]];
    }
    inputOffset += blockDim.x * inputStride;
    outputOffset += outputStride * blockDim.x;
  }
}

template <typename T>
void index_select_parrel(Tensor<T> *out, Tensor<T> *input, Tensor<int> *indices,
                         const int Dim, const int wchDim, const int idxLen) {
  if (wchDim == 0) {
    dim3 grid(idxLen);
    dim3 block(32);
    index_select_cuda_first<T><<<grid, block>>>(out, input, indices);
    // printf("cuda first\n");
  } else if (wchDim + 1 == Dim) {
    dim3 grid(32);
    dim3 block(32);
    index_select_cuda_last<T><<<grid, block>>>(out, input, indices);
    // printf("cuda last\n");
  } else {
    dim3 grid(32);
    dim3 block(32);
    index_select_cuda_mid<T><<<grid, block>>>(out, input, indices, wchDim);
    // printf("cuda mid\n");
  }
  return;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("usage: ./main [wchDim] [width] [len]\n");
    exit(0);
  }

  int wchDim = atoi(argv[1]);
  int Width = atoi(argv[2]);
  int len = atoi(argv[3]);
  const int ndim = 4;

  if (Width <= 0) {
    printf("Width > 0\n");
    exit(0);
  }

  if (wchDim >= 4 || wchDim < 0) {
    printf("0 <= wchDim < 4\n");
    exit(0);
  }

  if (len <= 0) {
    printf("len > 0\n");
    exit(0);
  }

  Tensor<int> *cTensor = (Tensor<int> *)malloc(sizeof(Tensor<int>));
  cTensor->shape = (int *)malloc(sizeof(int) * ndim);
  cTensor->shape[0] = Width;
  cTensor->shape[1] = Width;
  cTensor->shape[2] = Width;
  cTensor->shape[3] = Width;

  cTensor->stride = (int *)malloc(sizeof(int) * ndim);

  cTensor->stride[3] = 1;
  cTensor->stride[2] = cTensor->stride[3] * cTensor->shape[3];
  cTensor->stride[1] = cTensor->stride[2] * cTensor->shape[2];
  cTensor->stride[0] = cTensor->stride[1] * cTensor->shape[1];

  cTensor->ndim = ndim;

  const int sumElements = cTensor->stride[0] * cTensor->shape[0];
  cTensor->data = (int *)malloc(sumElements * sizeof(int));
  for (int ele = 0; ele < sumElements; ele++) {
    cTensor->data[ele] = ele;
  }
  printf("src Tensor sumElements %d shape (%d,%d,%d,%d) stride (%d,%d,%d,%d)\n",
         sumElements, cTensor->shape[0], cTensor->shape[1], cTensor->shape[2],
         cTensor->shape[3], cTensor->stride[0], cTensor->stride[1],
         cTensor->stride[2], cTensor->stride[3]);

  Tensor<int> *cgTensor = (Tensor<int> *)malloc(sizeof(Tensor<int>));

  cgTensor->ndim = cTensor->ndim;

  checkCudaErrors(cudaMalloc(&cgTensor->shape, sizeof(int) * ndim));
  checkCudaErrors(cudaMemcpy(cgTensor->shape, cTensor->shape,
                             sizeof(int) * ndim, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&cgTensor->stride, sizeof(int) * ndim));
  checkCudaErrors(cudaMemcpy(cgTensor->stride, cTensor->stride,
                             sizeof(int) * ndim, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&cgTensor->data, sizeof(int) * sumElements));
  checkCudaErrors(cudaMemcpy(cgTensor->data, cTensor->data,
                             sizeof(int) * sumElements,
                             cudaMemcpyHostToDevice));

  Tensor<int> *gTensor;

  checkCudaErrors(cudaMalloc(&gTensor, sizeof(Tensor<int>)));
  checkCudaErrors(cudaMemcpy(gTensor, cgTensor, sizeof(Tensor<int>),
                             cudaMemcpyHostToDevice));

  printf("gen gTensor src ok\n");

  Tensor<int> *cIndices = (Tensor<int> *)malloc(sizeof(Tensor<int>));
  cIndices->shape = (int *)malloc(sizeof(int));
  cIndices->shape[0] = len;
  cIndices->ndim = 1;

  cIndices->stride = (int *)malloc(sizeof(int));
  cIndices->stride[0] = 1;

  cIndices->data = (int *)malloc(len * sizeof(int));
  for (int i = 0; i < len; i++) {
    cIndices->data[i] = rand() % Width;
  }

  Tensor<int> *cgIndices = (Tensor<int> *)malloc(sizeof(Tensor<int>));
  cgIndices->ndim = 1;
  checkCudaErrors(cudaMalloc(&cgIndices->shape, sizeof(int)));
  checkCudaErrors(cudaMemcpy(cgIndices->shape, cIndices->shape, sizeof(int),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&cgIndices->stride, sizeof(int)));
  checkCudaErrors(cudaMemcpy(cgIndices->stride, cIndices->stride, sizeof(int),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&cgIndices->data, sizeof(int) * len));
  checkCudaErrors(cudaMemcpy(cgIndices->data, cIndices->data, sizeof(int) * len,
                             cudaMemcpyHostToDevice));

  Tensor<int> *gIndices;
  checkCudaErrors(cudaMalloc(&gIndices, sizeof(Tensor<int>)));
  checkCudaErrors(cudaMemcpy(gIndices, cgIndices, sizeof(Tensor<int>),
                             cudaMemcpyHostToDevice));
  printf("gIndices complete ok\n");

  Tensor<int> *cOut = (Tensor<int> *)malloc(sizeof(Tensor<int>));
  cOut->ndim = ndim;
  cOut->shape = (int *)malloc(sizeof(int) * ndim);
  for (int i = 0; i < cOut->ndim; i++) {
    cOut->shape[i] = cTensor->shape[i];
  }
  cOut->shape[wchDim] = len;

  cOut->stride = (int *)malloc(sizeof(int) * ndim);
  for (int i = cOut->ndim - 1; i >= 0; i--) {
    if (i + 1 == cOut->ndim) {
      cOut->stride[i] = 1;
    } else {
      cOut->stride[i] = cOut->stride[i + 1] * cOut->shape[i + 1];
    }
  }
  const int outElements = cOut->stride[0] * cOut->shape[0];
  cOut->data = (int *)malloc(sizeof(int) * outElements);

  printf("gen cOut out ok with outElements %d\n", outElements);
  Tensor<int> *cgOut = (Tensor<int> *)malloc(sizeof(Tensor<int>));
  cgOut->ndim = ndim;
  checkCudaErrors(cudaMalloc(&cgOut->shape, sizeof(int) * ndim));
  checkCudaErrors(cudaMemcpy(cgOut->shape, cOut->shape,
                             sizeof(int) * cOut->ndim, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&cgOut->stride, sizeof(int) * ndim));
  checkCudaErrors(cudaMemcpy(cgOut->stride, cOut->stride,
                             sizeof(int) * cOut->ndim, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&cgOut->data, sizeof(int) * outElements));

  Tensor<int> *gOut;
  checkCudaErrors(cudaMalloc(&gOut, sizeof(Tensor<int>)));
  checkCudaErrors(
      cudaMemcpy(gOut, cgOut, sizeof(cgOut), cudaMemcpyHostToDevice));
  printf("gout complete ok\n");

  float milliseconds = 0;
  cudaEvent_t start, stop;
  (cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));
  for (int i = 0; i < 100; i++) {
    index_select_parrel<int>(gOut, gTensor, gIndices, ndim, wchDim, len);
  }
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("record and run complete %f ms\n", milliseconds);
  printf("index_select_parrel ok\n");
  checkCudaErrors(cudaMemcpy(cOut->data, cgOut->data, sizeof(int) * outElements,
                             cudaMemcpyDeviceToHost));

  printf("cal ok\n");

  return 0;
}
