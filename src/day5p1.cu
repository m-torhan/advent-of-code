#include "utils.cuh"
#include <iostream>
#include <sstream>
#include <vector>

constexpr int N = 1024;
constexpr int threadsPerBlock = 256;
constexpr int blocksPerGrid = std::min(32, 1 + (N - 1) / threadsPerBlock);

__device__ bool check_update(const int *ordering, int orderingCount, const int *update, int updateLen) {
    bool ret = true;

    for (int i = 0; i < updateLen - 1; ++i) {
        for (int j = i + 1; j < updateLen; ++j) {
            for (int k = 0; k < orderingCount; ++k) {
                if (update[i] == ordering[2 * k + 1] && update[j] == ordering[2 * k]) {
                    ret = false;
                }
            }
        }
    }
    return ret;
}

__global__ void check_updates(const int *ordering, int orderingCount, const int *updates, int updatesCount,
                              const int *updatesLen, int maxUpdateLen, int *result) {
    __shared__ int cache[threadsPerBlock];

    int cacheIdx = threadIdx.x;
    int updateIdx = threadIdx.x + blockIdx.x * blockDim.x;

    int partialResult = 0;

    while (updateIdx < updatesCount) {
        if (check_update(ordering, orderingCount, &updates[updateIdx * maxUpdateLen], updatesLen[updateIdx])) {
            partialResult += updates[updateIdx * maxUpdateLen + updatesLen[updateIdx] / 2];
        }
        updateIdx += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = partialResult;

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
        result[blockIdx.x] = cache[0];
    }
}

int main() {
    std::vector<std::vector<int>> orderingVec;

    std::string line;

    while (std::getline(std::cin, line)) {
        if (line.find('|') == std::string::npos) {
            break;
        }
        std::stringstream sLine(line);

        std::vector<int> order;
        std::string s;
        while (std::getline(sLine, s, '|')) {
            order.push_back(std::stoi(s));
        }
        orderingVec.push_back(order);
    }

    std::vector<std::vector<int>> updatesVec;
    size_t maxUpdateLen = 0;

    while (std::getline(std::cin, line)) {
        std::stringstream sLine(line);

        std::vector<int> update;
        std::string s;
        while (std::getline(sLine, s, ',')) {
            update.push_back(std::stoi(s));
        }
        updatesVec.push_back(update);
        maxUpdateLen = std::max(maxUpdateLen, update.size());
    }

    const auto orderingCount = orderingVec.size();
    const auto updatesCount = updatesVec.size();

    int *ordering = new int[orderingCount * 2];
    int *updates = new int[updatesCount * maxUpdateLen];
    int *updatesLen = new int[updatesCount];
    int *partialResult = new int[blocksPerGrid];

    int *dev_ordering, *dev_updates, *dev_updatesLen, *dev_partialResult;

    for (int i = 0; i < orderingCount; ++i) {
        ordering[2 * i] = orderingVec[i][0];
        ordering[2 * i + 1] = orderingVec[i][1];
    }

    for (int i = 0; i < updatesCount; ++i) {
        memcpy(&updates[i * maxUpdateLen], updatesVec[i].data(), updatesVec[i].size() * sizeof(int));
        updatesLen[i] = updatesVec[i].size();
    }

    CUDA_CHECK(cudaMalloc((void **)&dev_ordering, orderingCount * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_updates, updatesCount * maxUpdateLen * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_updatesLen, updatesCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_partialResult, blocksPerGrid * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dev_ordering, ordering, orderingCount * 2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_updates, updates, updatesCount * maxUpdateLen * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_updatesLen, updatesLen, updatesCount * sizeof(int), cudaMemcpyHostToDevice));

    check_updates<<<blocksPerGrid, threadsPerBlock>>>(dev_ordering, orderingCount, dev_updates, updatesCount,
                                                      dev_updatesLen, maxUpdateLen, dev_partialResult);

    CUDA_CHECK(cudaMemcpy(partialResult, dev_partialResult, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));

    int result = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        result += partialResult[i];
    }

    std::cout << "Result: " << result << std::endl;

    cudaFree(dev_ordering);
    cudaFree(dev_updates);
    cudaFree(dev_updatesLen);
    cudaFree(dev_partialResult);

    delete[] ordering;
    delete[] updates;
    delete[] updatesLen;
    delete[] partialResult;
}
