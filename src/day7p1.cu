#include "utils.cuh"
#include <iostream>
#include <sstream>
#include <vector>

constexpr int N = 1024;
constexpr int threadsPerBlock = 256;
constexpr int blocksPerGrid = std::min(32, 1 + (N - 1) / threadsPerBlock);

__device__ bool is_equation_solvable(unsigned long long *equation, int equationLen) {
    bool ret = false;
    unsigned long long numCombinations = 2 << (equationLen - 2);

    for (unsigned long long i = 0; i < numCombinations; ++i) {
        unsigned long long result = equation[1];
        for (int j = 2; j < equationLen; ++j) {
            if ((i >> (j - 2)) & 1) {
                result += equation[j];
            } else {
                result *= equation[j];
            }
        }
        if (result == equation[0]) {
            ret = true;
        }
    }
    return ret;
}

__global__ void are_equations_solvable(unsigned long long *equations, int equationsCount, int *equationsLen,
                                       int maxEquationLen, unsigned long long *solvable) {
    __shared__ unsigned long long cache[threadsPerBlock];

    int equationIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    unsigned long long partial = 0;
    while (equationIdx < equationsCount) {
        if (is_equation_solvable(&equations[equationIdx * maxEquationLen], equationsLen[equationIdx])) {
            partial += equations[equationIdx * maxEquationLen];
        }
        equationIdx += blockDim.x * gridDim.x;
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
        solvable[blockIdx.x] = cache[0];
    }
}

int main() {
    std::vector<std::vector<unsigned long long>> equationsVec;

    size_t maxEquationLen = 0;

    for (std::string line; std::getline(std::cin, line);) {
        std::stringstream sline(line);
        unsigned long long num;
        char c;
        equationsVec.push_back({});

        sline >> num;
        equationsVec.back().push_back(num);
        sline >> c;

        while (sline >> num) {
            equationsVec.back().push_back(num);
        }
        maxEquationLen = std::max(maxEquationLen, equationsVec.back().size());
    }

    const auto equationsNum = equationsVec.size();

    unsigned long long *equations = new unsigned long long[maxEquationLen * equationsNum];
    int *equationsLen = new int[equationsNum];

    for (int i = 0; i < equationsNum; ++i) {
        equationsLen[i] = equationsVec[i].size();
        memcpy(&equations[i * maxEquationLen], equationsVec[i].data(),
               equationsVec[i].size() * sizeof(unsigned long long));
    }

    unsigned long long *partialResult = new unsigned long long[blocksPerGrid];
    int *dev_equationsLen;
    unsigned long long *dev_equations, *dev_partialResult;

    CUDA_CHECK(cudaMalloc((void **)&dev_equations, maxEquationLen * equationsNum * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void **)&dev_equationsLen, equationsNum * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_partialResult, blocksPerGrid * sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(dev_equations, equations, maxEquationLen * equationsNum * sizeof(unsigned long long),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_equationsLen, equationsLen, equationsNum * sizeof(int), cudaMemcpyHostToDevice));

    are_equations_solvable<<<blocksPerGrid, threadsPerBlock>>>(dev_equations, equationsNum, dev_equationsLen,
                                                               maxEquationLen, dev_partialResult);

    CUDA_CHECK(cudaMemcpy(partialResult, dev_partialResult, blocksPerGrid * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    unsigned long long sum = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        sum += partialResult[i];
    }

    std::cout << "Solvable equations sum: " << sum << std::endl;

    cudaFree(dev_equations);
    cudaFree(dev_equationsLen);
    cudaFree(dev_partialResult);

    delete[] equations;
    delete[] equationsLen;
    delete[] partialResult;
}
