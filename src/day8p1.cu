#include "utils.cuh"
#include <iostream>
#include <vector>

constexpr int N = 32;
constexpr dim3 blockDim(16, 16);
constexpr dim3 gridDim(N / blockDim.x, N / blockDim.y);
constexpr int blocksPerGrid = gridDim.x * gridDim.y;

__global__ void find_antinodes(const char *antennas, const int *antennasPos, int antennasCount, int mapShapeX,
                               int mapShapeY, int *antinodes) {
    for (int x = threadIdx.x + blockIdx.x * blockDim.x; x < antennasCount; x += blockDim.x * gridDim.x) {
        for (int y = threadIdx.y + blockIdx.y * blockDim.y; y < antennasCount; y += blockDim.y * gridDim.y) {
            if (x < y && antennas[x] == antennas[y]) {
                int deltaX = antennasPos[2 * y] - antennasPos[2 * x];
                int deltaY = antennasPos[2 * y + 1] - antennasPos[2 * x + 1];

                int antinode1X = antennasPos[2 * y] + deltaX;
                int antinode1Y = antennasPos[2 * y + 1] + deltaY;
                if (0 <= antinode1X && antinode1X < mapShapeX && 0 <= antinode1Y && antinode1Y < mapShapeY) {
                    antinodes[antinode1X + antinode1Y * mapShapeX] = 1;
                }

                int antinode2X = antennasPos[2 * x] - deltaX;
                int antinode2Y = antennasPos[2 * x + 1] - deltaY;

                if (0 <= antinode2X && antinode2X < mapShapeX && 0 <= antinode2Y && antinode2Y < mapShapeY) {
                    antinodes[antinode2X + antinode2Y * mapShapeX] = 1;
                }
            }
        }
    }
}
__global__ void count_antinodes(int *antinodes, int mapShapeX, int mapShapeY, int *count) {
    __shared__ int cache[blockDim.x * blockDim.y];

    int cacheIdx = threadIdx.x + threadIdx.y * blockDim.x;

    int partialCount = 0;

    for (int x = threadIdx.x + blockIdx.x * blockDim.x; x < mapShapeX; x += blockDim.x * gridDim.x) {
        for (int y = threadIdx.y + blockIdx.y * blockDim.y; y < mapShapeY; y += blockDim.y * gridDim.y) {
            partialCount += antinodes[x + y * mapShapeX];
        }
    }
    cache[cacheIdx] = partialCount;

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
        count[blockIdx.x + blockIdx.y * gridDim.x] = cache[0];
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

    std::vector<char> antennasVec;
    std::vector<std::pair<int, int>> antennasPosVec;

    for (int x = 0; x < mapShape[0]; ++x) {
        for (int y = 0; y < mapShape[1]; ++y) {
            char freq = mapVec[y][x];
            if (freq != '.') {
                antennasVec.push_back(freq);
                antennasPosVec.push_back({x, y});
            }
        }
    }

    const int antennasCount = antennasVec.size();
    int *antennasPos = new int[antennasCount * 2];
    int *partialCount = new int[blocksPerGrid];

    for (int i = 0; i < antennasCount; ++i) {
        antennasPos[2 * i] = antennasPosVec[i].first;
        antennasPos[2 * i + 1] = antennasPosVec[i].second;
    }

    char *dev_antennas;
    int *dev_antennasPos;
    int *dev_antinodes;
    int *dev_partialCount;

    CUDA_CHECK(cudaMalloc((void **)&dev_antennas, antennasCount * sizeof(char)));
    CUDA_CHECK(cudaMalloc((void **)&dev_antennasPos, antennasCount * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_antinodes, mapSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_partialCount, blocksPerGrid * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dev_antennas, antennasVec.data(), antennasCount * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_antennasPos, antennasPos, 2 * antennasCount * sizeof(int), cudaMemcpyHostToDevice));

    find_antinodes<<<gridDim, blockDim>>>(dev_antennas, dev_antennasPos, antennasCount, mapShape[0], mapShape[1],
                                          dev_antinodes);

    count_antinodes<<<gridDim, blockDim>>>(dev_antinodes, mapShape[0], mapShape[1], dev_partialCount);

    CUDA_CHECK(cudaMemcpy(partialCount, dev_partialCount, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));

    int count = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        count += partialCount[i];
    }

    std::cout << "Antinodes count: " << count << std::endl;

    cudaFree(dev_antennas);
    cudaFree(dev_antennasPos);
    cudaFree(dev_antinodes);
    cudaFree(dev_partialCount);

    delete[] antennasPos;
    delete[] partialCount;
}
