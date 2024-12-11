#include "utils.cuh"
#include <iostream>
#include <vector>

constexpr int N = 1024;
constexpr dim3 blockDim(16, 16);
constexpr dim3 gridDim(N / blockDim.x, N / blockDim.y);
constexpr int blocksPerGrid = gridDim.x * gridDim.y;

__global__ void similarity_score(int *a, int *b, int *c, size_t size) {
    __shared__ int cache[blockDim.x * blockDim.y];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int cacheIdx = threadIdx.x + threadIdx.y * blockDim.x;

    int partial = 0;
    while (x < size) {
        while (y < size) {
            if (a[x] == b[y]) {
                partial += a[x];
            }
            y += blockDim.y * gridDim.y;
        }
        x += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = partial;

    __syncthreads();

    int idx = (blockDim.x * blockDim.y) / 2;
    while (idx != 0) {
        if (cacheIdx < idx) {
            cache[cacheIdx] += cache[cacheIdx + idx];
        }
        __syncthreads();
        idx /= 2;
    }

    if (cacheIdx == 0) {
        c[blockIdx.x + blockIdx.y * gridDim.x] = cache[0];
    }
}

int main() {
    std::vector<int> left;
    std::vector<int> right;

    int leftNum;
    int rightNum;

    while (std::cin >> leftNum >> rightNum) {
        left.push_back(leftNum);
        right.push_back(rightNum);
    }

    const auto vecLength = left.size();

    auto *partialScore = new int[blocksPerGrid];
    int *dev_left, *dev_right, *dev_partialScore;

    CUDA_CHECK(cudaMalloc((void **)&dev_left, vecLength * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_right, vecLength * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_partialScore, blocksPerGrid * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dev_left, left.data(), vecLength * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_right, right.data(), vecLength * sizeof(int), cudaMemcpyHostToDevice));

    similarity_score<<<gridDim, blockDim>>>(dev_left, dev_right, dev_partialScore, vecLength);

    CUDA_CHECK(cudaMemcpy(partialScore, dev_partialScore, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));

    int score = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        score += partialScore[i];
    }

    std::cout << "Similarity score: " << score << std::endl;

    cudaFree(dev_left);
    cudaFree(dev_right);
    cudaFree(dev_partialScore);

    delete[] partialScore;
}
