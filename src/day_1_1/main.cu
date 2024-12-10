#include "utils.cuh"
#include <iostream>
#include <queue>

constexpr int N = 1024;
constexpr int threadsPerBlock = 256;
constexpr int blocksPerGrid = std::min(32, 1 + (N - 1) / threadsPerBlock);

__global__ void total_dist(int *a, int *b, int *c, size_t size) {
    __shared__ int cache[threadsPerBlock];

    int vecIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    int partial = 0;
    while (vecIdx < size) {
        partial += abs(a[vecIdx] - b[vecIdx]);
        vecIdx += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = partial;

    __syncthreads();

    int idx = blockDim.x / 2;
    while (idx != 0) {
        if (cacheIdx < idx) {
            cache[cacheIdx] += cache[cacheIdx + idx];
        }
        __syncthreads();
        idx /= 2;
    }

    if (cacheIdx == 0) {
        c[blockIdx.x] = cache[0];
    }
}

int main() {
    std::priority_queue<int> left_q;
    std::priority_queue<int> right_q;

    int left_num;
    int right_num;

    while (std::cin >> left_num >> right_num) {
        left_q.push(left_num);
        right_q.push(right_num);
    }

    const auto vec_length = left_q.size();

    auto *left = new int[vec_length];
    auto *right = new int[vec_length];
    auto *partial_dist = new int[blocksPerGrid];
    int *dev_left, *dev_right, *dev_partial_dist;

    for (int i = 0; i < vec_length; ++i) {
        left[i] = left_q.top();
        right[i] = right_q.top();
        left_q.pop();
        right_q.pop();
    }

    CUDA_CHECK(cudaMalloc((void **)&dev_left, vec_length * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_right, vec_length * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_partial_dist, blocksPerGrid * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dev_left, left, vec_length * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_right, right, vec_length * sizeof(int), cudaMemcpyHostToDevice));

    total_dist<<<blocksPerGrid, threadsPerBlock>>>(dev_left, dev_right, dev_partial_dist, vec_length);

    CUDA_CHECK(cudaMemcpy(partial_dist, dev_partial_dist, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));

    int dist = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        dist += partial_dist[i];
    }

    std::cout << "Total distance: " << dist << std::endl;

    cudaFree(dev_left);
    cudaFree(dev_right);
    cudaFree(dev_partial_dist);

    delete[] left;
    delete[] right;
    delete[] partial_dist;
}
