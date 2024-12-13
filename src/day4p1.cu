#include "utils.cuh"
#include <iostream>
#include <vector>

constexpr int N = 32;
constexpr dim3 blockDim(16, 16);
constexpr dim3 gridDim(N / blockDim.x, N / blockDim.y);
constexpr int blocksPerGrid = gridDim.x * gridDim.y;

// clang-format off
__device__ char kernels4x4[4][4][4] = {
    {
        {'X', '.', '.', '.'},
        {'.', 'M', '.', '.'},
        {'.', '.', 'A', '.'},
        {'.', '.', '.', 'S'},
    },
    {
        {'S', '.', '.', '.'},
        {'.', 'A', '.', '.'},
        {'.', '.', 'M', '.'},
        {'.', '.', '.', 'X'},
    },
    {
        {'.', '.', '.', 'X'},
        {'.', '.', 'M', '.'},
        {'.', 'A', '.', '.'},
        {'S', '.', '.', '.'},
    },
    {
        {'.', '.', '.', 'S'},
        {'.', '.', 'A', '.'},
        {'.', 'M', '.', '.'},
        {'X', '.', '.', '.'},
    },
};

__device__ char kernels4x1[2][1][4] = {
    {
        {'X', 'M', 'A', 'S'}
    },
    {
        {'S', 'A', 'M', 'X'}
    },
};

__device__ char kernels1x4[2][4][1] = {
    {
        {'X'},
        {'M'},
        {'A'},
        {'S'},
    },
    {
        {'S'},
        {'A'},
        {'M'},
        {'X'},
    },
};

// clang-format on

template <int KERNEL_W, int KERNEL_H>
__device__ int find_words_local(char *letters, int *lettersShape, const char kernel[KERNEL_H][KERNEL_W]) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int count = 0;
    while (x + KERNEL_W < lettersShape[1] + 1) {
        while (y + KERNEL_H < lettersShape[0] + 1) {
            int equals = 0;
            for (int i = 0; i < KERNEL_W; ++i) {
                for (int j = 0; j < KERNEL_H; ++j) {
                    if (kernel[j][i] != '.' && kernel[j][i] == letters[x + i + lettersShape[1] * (y + j)]) {
                        ++equals;
                    }
                }
            }
            if (equals == 4) {
                ++count;
            }
            y += blockDim.y * gridDim.y;
        }
        x += blockDim.x * gridDim.x;
        y = threadIdx.y + blockIdx.y * blockDim.y;
    }

    return count;
}

__global__ void find_words(char *letters, int *lettersShape, int *count) {
    __shared__ int cache[blockDim.x * blockDim.y];

    int cacheIdx = threadIdx.x + threadIdx.y * blockDim.x;
    int partialCount = 0;

    for (int i = 0; i < std::size(kernels4x4); ++i) {
        partialCount += find_words_local<4, 4>(letters, lettersShape, kernels4x4[i]);
    }
    for (int i = 0; i < std::size(kernels4x1); ++i) {
        partialCount += find_words_local<4, 1>(letters, lettersShape, kernels4x1[i]);
    }
    for (int i = 0; i < std::size(kernels1x4); ++i) {
        partialCount += find_words_local<1, 4>(letters, lettersShape, kernels1x4[i]);
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
    std::vector<std::vector<char>> lettersVec;

    std::string line;

    while (std::getline(std::cin, line)) {
        std::vector<char> lineVec(line.begin(), line.end());
        lettersVec.push_back(lineVec);
    }

    const int N = lettersVec.size();
    const int M = lettersVec[0].size();

    auto *letters = new char[N * M];
    int lettersShape[2] = {N, M};
    auto *partialCount = new int[blocksPerGrid];

    for (int i = 0; i < N; ++i) {
        memcpy(&letters[i * M], lettersVec[i].data(), M * sizeof(char));
    }

    char *dev_letters;
    int *dev_lettersShape;
    int *dev_partialCount;

    CUDA_CHECK(cudaMalloc((void **)&dev_letters, N * M * sizeof(char)));
    CUDA_CHECK(cudaMalloc((void **)&dev_lettersShape, 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_partialCount, blocksPerGrid * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dev_letters, letters, N * M * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_lettersShape, lettersShape, 2 * sizeof(int), cudaMemcpyHostToDevice));

    find_words<<<gridDim, blockDim>>>(dev_letters, dev_lettersShape, dev_partialCount);

    CUDA_CHECK(cudaMemcpy(partialCount, dev_partialCount, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));

    int count = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        count += partialCount[i];
    }

    std::cout << "Word count: " << count << std::endl;

    cudaFree(dev_letters);
    cudaFree(dev_lettersShape);
    cudaFree(dev_partialCount);

    delete[] letters;
    delete[] partialCount;
}
