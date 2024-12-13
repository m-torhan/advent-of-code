#include "utils.cuh"
#include <iostream>
#include <vector>

constexpr int N = 32;
constexpr dim3 blockDim(16, 16);
constexpr dim3 gridDim(N / blockDim.x, N / blockDim.y);
constexpr int blocksPerGrid = gridDim.x * gridDim.y;

__global__ void find_guard(char *map, int *mapShape, int *guardPos) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int pos[2];
    bool found = false;

    while (x < mapShape[1]) {
        while (y < mapShape[0]) {
            if (('^' == map[x + mapShape[1] * y]) || ('>' == map[x + mapShape[1] * y]) ||
                ('v' == map[x + mapShape[1] * y]) || ('<' == map[x + mapShape[1] * y])) {
                pos[0] = x;
                pos[1] = y;
                found = true;
            }
            y += blockDim.y * gridDim.y;
        }
        x += blockDim.x * gridDim.x;
        y = threadIdx.y + blockIdx.y * blockDim.y;
    }

    __syncthreads();

    if (found) {
        guardPos[0] = pos[0];
        guardPos[1] = pos[1];
    }
}

enum class Dir {
    UP,
    RIGHT,
    LEFT,
    DOWN,
};

int move_guard(char *map, const int mapShape[2], const int guardPos[2]) {
    int x = guardPos[0];
    int y = guardPos[1];

    std::vector<std::pair<int, int>> turns;

    Dir d;

    switch (map[x + y * mapShape[1]]) {
    case '^':
        d = Dir::UP;
        break;
    case '>':
        d = Dir::RIGHT;
        break;
    case 'v':
        d = Dir::DOWN;
        break;
    case '<':
        d = Dir::LEFT;
        break;
    }

    while (0 <= x && x <= mapShape[0] && 0 <= y && y <= mapShape[1]) {
        int x_next = x;
        int y_next = y;
        switch (d) {
        case Dir::UP:
            --y_next;
            break;
        case Dir::RIGHT:
            ++x_next;
            break;
        case Dir::DOWN:
            ++y_next;
            break;
        case Dir::LEFT:
            --x_next;
            break;
        }
        if (x_next < 0 || x_next >= mapShape[0] || y_next < 0 || y_next >= mapShape[1]) {
            break;
        }
        if (map[x_next + y_next * mapShape[1]] != '#') {
            x = x_next;
            y = y_next;
        } else {
            if (turns.size() > 1) {
                for (int i = 0; i < turns.size() - 1; ++i) {
                    if (turns[i].first == x && turns[i].second == y) {
                        return 1;
                    }
                }
            }
            turns.push_back({x, y});

            switch (d) {
            case Dir::UP:
                d = Dir::RIGHT;
                break;
            case Dir::RIGHT:
                d = Dir::DOWN;
                break;
            case Dir::DOWN:
                d = Dir::LEFT;
                break;
            case Dir::LEFT:
                d = Dir::UP;
                break;
            }
        }
    }

    return 0;
}

int main() {
    std::vector<std::vector<char>> mapVec;

    std::string line;

    while (std::getline(std::cin, line)) {
        std::vector<char> lineVec(line.begin(), line.end());
        mapVec.push_back(lineVec);
    }

    int mapShape[2] = {(int)mapVec.size(), (int)mapVec[0].size()};
    int mapSize = mapShape[0] * mapShape[1];
    char *map = new char[mapSize];
    int guardPos[2];
    int *partialCount = new int[blocksPerGrid];

    for (int i = 0; i < mapShape[1]; ++i) {
        memcpy(&map[i * mapShape[0]], mapVec[i].data(), mapShape[1] * sizeof(char));
    }

    char *dev_map;
    int *dev_mapShape;
    int *dev_guardPos;
    int *dev_partialCount;

    CUDA_CHECK(cudaMalloc((void **)&dev_map, mapSize * sizeof(char)));
    CUDA_CHECK(cudaMalloc((void **)&dev_mapShape, 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_guardPos, 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_partialCount, blocksPerGrid * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dev_map, map, mapSize * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_mapShape, mapShape, 2 * sizeof(int), cudaMemcpyHostToDevice));

    find_guard<<<gridDim, blockDim>>>(dev_map, dev_mapShape, dev_guardPos);

    CUDA_CHECK(cudaMemcpy(guardPos, dev_guardPos, 2 * sizeof(int), cudaMemcpyDeviceToHost));

    int loops = 0;
    char *tmpMap = new char[mapSize];

    for (int x = 0; x < mapShape[0]; ++x) {
        for (int y = 0; y < mapShape[1]; ++y) {
            memcpy(tmpMap, map, mapSize * sizeof(char));
            int mapIdx = x + y * mapShape[1];
            if ('.' == tmpMap[mapIdx]) {
                tmpMap[mapIdx] = '#';
                loops += move_guard(tmpMap, mapShape, guardPos);
            }
        }
    }

    std::cout << "Loop count: " << loops << std::endl;

    cudaFree(dev_map);
    cudaFree(dev_mapShape);
    cudaFree(dev_partialCount);
    cudaFree(dev_guardPos);

    delete[] map;
    delete[] tmpMap;
    delete[] partialCount;
}
