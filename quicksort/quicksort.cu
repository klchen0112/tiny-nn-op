#include <algorithm>


#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

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
constexpr int MAX_DEPTH = 16;
constexpr int INSERTION_SORT = 32;

__device__ void selection_sort(float *data, int left, int right) {
  // printf("selection sort from %d to %d\n",left,right);
  for (int i = left; i < right; i++) {
    float min_val = data[i];
    int min_idx = i;
    for (int j = i + 1; j <= right; j++) {
      if (data[j] < min_val) {
        min_idx = j;
        min_val = data[j];
      }
    }
    if (i != min_idx) {
      data[min_idx] = data[i];
      data[i] = min_val;
    }
  }
}

__global__ void quick_sort(float *data, int left, int right, int depth) {
  if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT) {
    selection_sort(data, left, right);
    return;
  }
  // printf("DEPTH %d\n",depth);

  float *left_ptr = data + left;
  float *right_ptr = data + right;

  float pivot = data[right];

  // partion

  while (left_ptr <= right_ptr) {
    float left_val = *left_ptr;
    float right_val = *right_ptr;
    while (left_val < pivot) {
      left_ptr++;
      left_val = *left_ptr;
    }

    while (right_val > pivot) {
      right_ptr--;
      right_val = *right_ptr;
    }

    if (left_ptr <= right_ptr) {
      *left_ptr = right_val;
      left_ptr++;
      *right_ptr = left_val;
      right_ptr--;
    }
  }

  int n_right = right_ptr - data;
  int n_left = left_ptr - data;

  if (left < (right_ptr - data)) {
    cudaStream_t l_stream;
    // 设置非阻塞流
    cudaStreamCreateWithFlags(&l_stream, cudaStreamNonBlocking);
    quick_sort<<<1, 1, 0, l_stream>>>(data, left, n_right, depth + 1);
    cudaStreamDestroy(l_stream);
  }

  if ((left_ptr - data) < right) {
    cudaStream_t r_stream;
    // 设置非阻塞流
    cudaStreamCreateWithFlags(&r_stream, cudaStreamNonBlocking);
    quick_sort<<<1, 1, 0, r_stream>>>(data, n_left, right, depth + 1);
    cudaStreamDestroy(r_stream);
  }
}

void run_qsort(float *data, int nitems) {
  // Prepare CDP for the max depth 'MAX_DEPTH'.
  checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));
  printf("run own qsort\n");
  int left = 0;
  int right = nitems - 1;
  quick_sort<<<1, 1>>>(data, left, right, 0);
  checkCudaErrors(cudaDeviceSynchronize());
}

int main() {
  float milliseconds;
  const int N = 5e4;

  float *c_data = (float *)malloc(N * sizeof(float));
  std::generate(c_data, c_data + N, rand);


  float *g_data = nullptr;

  checkCudaErrors(cudaMalloc(&g_data, N * sizeof(float)));
  checkCudaErrors(
      cudaMemcpy(g_data, c_data, N * sizeof(float), cudaMemcpyHostToDevice));

  thrust::device_ptr<float> t_data(g_data);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  thrust::sort( t_data, t_data + N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  float *t_result = (float *)malloc(N * sizeof(float));
  checkCudaErrors(
      cudaMemcpy(t_result, g_data, N * sizeof(float), cudaMemcpyDeviceToHost));
  printf("thrust sort latency = %f ms\n", milliseconds);

  checkCudaErrors(
      cudaMemcpy(g_data, c_data, N * sizeof(float), cudaMemcpyHostToDevice));

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  run_qsort(g_data, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("own sort latency = %f ms\n", milliseconds);


  float *g_result = (float *)malloc(N * sizeof(float));
  checkCudaErrors(
      cudaMemcpy(g_result, g_data, N * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    if (g_result[i] != t_result[i]) {
      printf("%d-th error", i);
      return 0;
    }
  }

  checkCudaErrors(cudaFree(g_data));
  free(c_data);
  free(g_result);
  return 0;
}
