#include "utils.cuh"
#include <assert.h>
#include <iostream>
#include <vector>

constexpr int N = 32;
constexpr dim3 blockDim(16, 16);
constexpr dim3 gridDim(N / blockDim.x, N / blockDim.y);
constexpr int blocksPerGrid = gridDim.x * gridDim.y;

__device__ int count_trails(const char *map, int mapShapeX, int mapShapeY, int startX, int startY) {
    int ret = 0;
    bool *visited = new bool[mapShapeX * mapShapeY];
    int *q = new int[4 * 10 * 2];
    int q_idx = 0;

    memset(visited, 0, mapShapeX * mapShapeY * sizeof(char));

    visited[startX + startY * mapShapeX] = true;
    q[2 * q_idx] = startX;
    q[2 * q_idx + 1] = startY;
    ++q_idx;

    while (q_idx > 0) {
        --q_idx;
        int x = q[2 * q_idx];
        int y = q[2 * q_idx + 1];

        const std::pair<int, int> nexts[4] = {{x - 1, y}, {x + 1, y}, {x, y - 1}, {x, y + 1}};

        for (auto &next : nexts) {
            const int nextX = next.first;
            const int nextY = next.second;

            if (0 <= nextX && nextX < mapShapeX && 0 <= nextY && nextY < mapShapeY &&
                !visited[nextX + nextY * mapShapeX] && map[nextX + nextY * mapShapeX] == map[x + y * mapShapeY] + 1) {
                visited[nextX + nextY * mapShapeX] = true;
                if (map[nextX + nextY * mapShapeX] == '9') {
                    ++ret;
                } else {
                    assert(q_idx < 4 * 10 * 2);
                    q[2 * q_idx] = nextX;
                    q[2 * q_idx + 1] = nextY;
                    ++q_idx;
                }
            }
        }
    }

    delete[] q;
    delete[] visited;

    return ret;
}

__global__ void count_all_trails(const char *map, int mapShapeX, int mapShapeY, int *result) {
    __shared__ int cache[blockDim.x * blockDim.y];

    int cacheIdx = threadIdx.x + threadIdx.y * blockDim.x;

    int partialResult = 0;

    for (int x = threadIdx.x + blockIdx.x * blockDim.x; x < mapShapeX; x += blockDim.x * gridDim.x) {
        for (int y = threadIdx.y + blockIdx.y * blockDim.y; y < mapShapeY; y += blockDim.y * gridDim.y) {
            if (map[x + y * mapShapeX] == '0') {
                partialResult += count_trails(map, mapShapeX, mapShapeY, x, y);
            }
        }
    }
    cache[cacheIdx] = partialResult;

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
        result[blockIdx.x + blockIdx.y * gridDim.x] = cache[0];
    }
}

int main() {
    std::vector<std::vector<char>> mapVec;

    std::string line;

    while (std::getline(std::cin, line)) {
        std::vector<char> lineVec(line.begin(), line.end());
        mapVec.push_back(lineVec);
    }

    int mapShape[2] = {(int)mapVec[0].size(), (int)mapVec.size()};
    int mapSize = mapShape[0] * mapShape[1];
    char *map = new char[mapSize];
    int *partialResult = new int[blocksPerGrid];

    for (int i = 0; i < mapVec.size(); ++i) {
        memcpy(&map[i * mapShape[0]], mapVec[i].data(), mapShape[0] * sizeof(char));
    }

    char *dev_map;
    int *dev_partialResult;

    CUDA_CHECK(cudaMalloc((void **)&dev_map, mapSize * sizeof(char)));
    CUDA_CHECK(cudaMalloc((void **)&dev_partialResult, blocksPerGrid * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dev_map, map, mapSize * sizeof(char), cudaMemcpyHostToDevice));

    count_all_trails<<<gridDim, blockDim>>>(dev_map, mapShape[0], mapShape[1], dev_partialResult);

    CUDA_CHECK(cudaMemcpy(partialResult, dev_partialResult, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));

    int count = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        count += partialResult[i];
    }

    std::cout << "Trails count: " << count << std::endl;

    cudaFree(dev_map);
    cudaFree(dev_partialResult);

    delete[] map;
    delete[] partialResult;
}
