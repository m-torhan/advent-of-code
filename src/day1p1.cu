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
    std::priority_queue<int> leftQ;
    std::priority_queue<int> rightQ;

    int leftNum;
    int rightNum;

    while (std::cin >> leftNum >> rightNum) {
        leftQ.push(leftNum);
        rightQ.push(rightNum);
    }

    const auto vecLength = leftQ.size();

    auto *left = new int[vecLength];
    auto *right = new int[vecLength];
    auto *partialDist = new int[blocksPerGrid];
    int *dev_left, *dev_right, *dev_partialDist;

    for (int i = 0; i < vecLength; ++i) {
        left[i] = leftQ.top();
        right[i] = rightQ.top();
        leftQ.pop();
        rightQ.pop();
    }

    CUDA_CHECK(cudaMalloc((void **)&dev_left, vecLength * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_right, vecLength * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_partialDist, blocksPerGrid * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dev_left, left, vecLength * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_right, right, vecLength * sizeof(int), cudaMemcpyHostToDevice));

    total_dist<<<blocksPerGrid, threadsPerBlock>>>(dev_left, dev_right, dev_partialDist, vecLength);

    CUDA_CHECK(cudaMemcpy(partialDist, dev_partialDist, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));

    int dist = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        dist += partialDist[i];
    }

    std::cout << "Total distance: " << dist << std::endl;

    cudaFree(dev_left);
    cudaFree(dev_right);
    cudaFree(dev_partialDist);

    delete[] left;
    delete[] right;
    delete[] partialDist;
}
