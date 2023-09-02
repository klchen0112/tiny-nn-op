#include "cuda_runtime.h"
#include <cmath>
#include <cstddef>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
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
__global__ void copy_cuda_first(Tensor<T> *out, Tensor<T> *(inputs[]),
                                const int nTensor) {

  // a block copy a tensor
  __shared__ int offset;
  const int wch_tensor = blockIdx.x;
  int tid = threadIdx.x;
  if (tid == 0) {
    offset = 0;
    for (int i = 0; i < wch_tensor; i++) {
      offset = offset + inputs[i]->shape[0] * inputs[i]->stride[0];
    }
    printf("block %d offset %d\n", wch_tensor, offset);
  }
  __syncthreads();
  const auto outOffset = offset;
  // all thread in block copy the value with stirde(numThreads)
  Tensor<T> *input = inputs[wch_tensor];
  const int nElements = input->shape[0] * input->stride[0];
  const int stride = gridDim.x;
  while (tid < nElements) {
    out->data[outOffset + tid] = input->data[tid];
    tid += stride;
  }
}

template <typename T>
__global__ void copy_cuda_mid(Tensor<T> *out, Tensor<T> *(inputs[]),
                              const int nTensor, const int wchDim) {
  const int wch_tensor = blockIdx.x;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // gridY process ()
  // gridDim.x tensor process (wchDim,...) tensor copy
  Tensor<T> *input = inputs[wch_tensor];

  __shared__ int in_offset, out_offset;
  // a grid cal once
  if (ty == 0 && tx == 0) {
    in_offset = 0;
    for (int i = 0; i < wch_tensor; i++) {
      in_offset += inputs[i]->stride[wchDim - 1];
    }
    out_offset = in_offset;
    for (int i = wch_tensor; i < nTensor; i++) {
      out_offset += inputs[i]->stride[wchDim - 1];
    }
    printf("tensor %d in_offset %d out_offset %d\n", wch_tensor, in_offset,
           out_offset);
  }
  __syncthreads();

  const int totalGrid =
      input->stride[0] / input->stride[wchDim - 1] * input->shape[0];
  const int elements = input->stride[wchDim - 1];
  for (int nth = ty; nth < totalGrid; nth += gridDim.y) {
    const int local_offset = nth * out_offset + in_offset;
    const int input_offset = elements * nth;
    for (int nEle = tx; nEle < elements; nEle += gridDim.x) {
      out->data[local_offset + nEle] = input->data[input_offset + nEle];
    }
  }
}

template <typename T>
__global__ void copy_cuda_last(Tensor<T> *out, Tensor<T> *(inputs[]),
                               const int nTensor) {
  // a block copy a tensor
  __shared__ int offset;
  // calculate thread offset
  __shared__ int needThread;
  const int wch_tensor = blockIdx.x;
  int tid = threadIdx.x;
  if (tid == 0) {
    offset = 0;
    for (int i = 0; i + 1 < wch_tensor; i++) {
      offset += inputs[i]->shape[0] * inputs[i]->stride[0];
    }
    needThread = inputs[wch_tensor]->shape[0] * inputs[wch_tensor]->stride[0] /
                 inputs[wch_tensor]->shape[nTensor - 1];
  }
  __syncthreads();

  Tensor<T> *input = inputs[wch_tensor];
  const int elements = input->shape[nTensor - 1];
  int dataOffset = elements * tid;
  for (int i = tid; i < needThread; i += gridDim.x) {
    // a thread copy a last dim
    for (int j = 0; j < elements; j++) {
      out->data[offset + dataOffset + j] = input->data[dataOffset + j];
    }
    dataOffset += elements * gridDim.x;
  }
}

template <typename T>
void concat_parrel(Tensor<T> *out, Tensor<T> *(inputs[]), const int nTensor,
                   const int Dim, const int wchDim) {

  // printf("concat %d tensor with %d Dim in %d wchDim\n",nTensor,Dim,wchDim);
  if (wchDim == 0) {
    dim3 grid(nTensor);
    dim3 block(32);
    copy_cuda_first<T><<<grid, block>>>(out, inputs, nTensor);
    // printf("cuda first\n");
  } else if (wchDim + 1 == Dim) {
    dim3 grid(nTensor);
    dim3 block(32);
    copy_cuda_last<T><<<grid, block>>>(out, inputs, nTensor);
    // printf("cuda last\n");
  } else {
    dim3 grid(nTensor);
    dim3 block(16, 16);
    copy_cuda_mid<T><<<grid, block>>>(out, inputs, nTensor, wchDim);
    // printf("cuda mid\n");
  }
  return;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("usage: ./main [wchDim] [width] [nTnesor]\n");
    exit(0);
  }

  int wchDim = atoi(argv[1]);
  int Width = atoi(argv[2]);
  int nTensor = atoi(argv[3]);
  const int ndim = 4;

  if (nTensor < 0) {
    printf("nTensor > 0\n");
    exit(0);
  }
  if (Width <= 0) {
    printf("Width > 0\n");
    exit(0);
  }

  if (wchDim >= 4 || wchDim < 0) {
    printf("0 <= wchDim < 4\n");
    exit(0);
  }

  Tensor<int> *(cTensor[nTensor]);

  Tensor<int> *(cgTensor[nTensor]);
  Tensor<int> *(gTensor[nTensor]);

  int outElements = 0;

  for (int i = 0; i < nTensor; i++) {
    cTensor[i] = (Tensor<int> *)malloc(sizeof(Tensor<int>));

    cTensor[i]->shape = (int *)malloc(sizeof(int) * ndim);
    cTensor[i]->shape[0] = 10;
    cTensor[i]->shape[1] = 3;
    cTensor[i]->shape[2] = Width;
    cTensor[i]->shape[3] = Width;
    cTensor[i]->shape[wchDim] = (i + 1);

    cTensor[i]->stride = (int *)malloc(sizeof(int) * ndim);

    cTensor[i]->stride[3] = 1;
    cTensor[i]->stride[2] = cTensor[i]->stride[3] * cTensor[i]->shape[3];
    cTensor[i]->stride[1] = cTensor[i]->stride[2] * cTensor[i]->shape[2];
    cTensor[i]->stride[0] = cTensor[i]->stride[1] * cTensor[i]->shape[1];

    cTensor[i]->ndim = ndim;

    const int sumElements = cTensor[i]->stride[0] * cTensor[i]->shape[0];
    outElements += sumElements;
    cTensor[i]->data = (int *)malloc(sumElements * sizeof(int));
    for (int ele = 0; ele < sumElements; ele++) {
      cTensor[i]->data[ele] = i;
    }

    printf("%d-th Tensor sumElements %d shape (%d,%d,%d,%d) stride "
           "(%d,%d,%d,%d)\n",
           i, sumElements, cTensor[i]->shape[0], cTensor[i]->shape[1],
           cTensor[i]->shape[2], cTensor[i]->shape[3], cTensor[i]->stride[0],
           cTensor[i]->stride[1], cTensor[i]->stride[2], cTensor[i]->stride[3]);
    // gen dst tensor

    cgTensor[i] = (Tensor<int> *)malloc(sizeof(Tensor<int>));
    cgTensor[i]->ndim = cTensor[i]->ndim;

    checkCudaErrors(cudaMalloc(&cgTensor[i]->shape, sizeof(int) * ndim));
    checkCudaErrors(cudaMemcpy(cgTensor[i]->shape, cTensor[i]->shape,
                               sizeof(int) * ndim, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&cgTensor[i]->stride, sizeof(int) * ndim));
    checkCudaErrors(cudaMemcpy(cgTensor[i]->stride, cTensor[i]->stride,
                               sizeof(int) * ndim, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&cgTensor[i]->data, sizeof(int) * sumElements));
    checkCudaErrors(cudaMemcpy(cgTensor[i]->data, cTensor[i]->data,
                               sizeof(int) * sumElements,
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&gTensor[i], sizeof(Tensor<int>)));
    checkCudaErrors(cudaMemcpy(gTensor[i], cgTensor[i], sizeof(Tensor<int>),
                               cudaMemcpyHostToDevice));
  }

  Tensor<int> **gsrc;

  checkCudaErrors(cudaMalloc(&gsrc, sizeof(Tensor<int> *) * nTensor));
  checkCudaErrors(cudaMemcpy(gsrc, gTensor, sizeof(Tensor<int> *) * nTensor,
                             cudaMemcpyHostToDevice));

  printf("gen gTensor src ok\n");

  // Gen Output
  Tensor<int> *cOut = (Tensor<int> *)malloc(sizeof(Tensor<int>));

  cOut->ndim = ndim;

  cOut->shape = (int *)malloc(sizeof(int) * ndim);
  cOut->shape[0] = 10;
  cOut->shape[1] = 3;
  cOut->shape[2] = Width;
  cOut->shape[3] = Width;
  cOut->shape[wchDim] = (1 + nTensor) * nTensor / 2;

  cOut->stride = (int *)malloc(sizeof(int) * ndim);
  cOut->stride[3] = 1;
  cOut->stride[2] = cOut->stride[3] * cOut->shape[3];
  cOut->stride[1] = cOut->stride[2] * cOut->shape[2];
  cOut->stride[0] = cOut->stride[1] * cOut->shape[1];
  printf("dst tensor shape (%d,%d,%d,%d)\n", cOut->shape[0], cOut->shape[1],
         cOut->shape[2], cOut->shape[3]);

  printf("OutELements %d\n", outElements);
  cOut->data = (int *)malloc(sizeof(int) * outElements);

  printf("gen cOut out ok\n");
  Tensor<int> *cgOut = (Tensor<int> *)malloc(sizeof(Tensor<int>));

  cgOut->ndim = ndim;
  checkCudaErrors(cudaMalloc(&cgOut->shape, sizeof(int) * ndim));
  checkCudaErrors(cudaMemcpy(cgOut->shape, cgOut->shape, sizeof(int) * ndim,
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&cgOut->stride, sizeof(int) * ndim));
  checkCudaErrors(cudaMemcpy(cgOut->stride, cgOut->stride, sizeof(int) * ndim,
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&cgOut->data, sizeof(int) * outElements));

  Tensor<int> *gOut;
  checkCudaErrors(cudaMalloc(&gOut, sizeof(Tensor<int>)));
  checkCudaErrors(
      cudaMemcpy(gOut, cgOut, sizeof(Tensor<int>), cudaMemcpyHostToDevice));

  printf("gout complete ok\n");
  float milliseconds = 0;
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));
  for (int i = 0; i < 1; i++) {
    concat_parrel<int>(gOut, gsrc, nTensor, ndim, wchDim);
  }
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("record and run complete %f ms\n", milliseconds);
  checkCudaErrors(cudaMemcpy(cOut->data, cgOut->data, sizeof(int) * outElements,
                             cudaMemcpyDeviceToHost));

  printf("%d\n", cOut->data[outElements - 1]);
  printf("cal ok\n");

  return 0;
}
