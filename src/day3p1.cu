#include "utils.cuh"
#include <iostream>

__device__ constexpr int WINDOW_W = 32;
constexpr int N = 1024;
constexpr int threadsPerBlock = 256;
constexpr int blocksPerGrid = std::min(32, 1 + (N - 1) / threadsPerBlock);

enum class ParserState {
    NONE,
    M,
    U,
    L,
    LP,
    NUM1,
    COMMA,
    NUM2,
    RP,
};

__device__ int check_program_window(const char *program, int programLen) {
    int ret = 0;
    ParserState state = ParserState::NONE;

    int num1 = 0;
    int num2 = 0;

    int i = 0;
    while (i < programLen) {
        switch (state) {
        case ParserState::NONE:
            if (i >= WINDOW_W / 2) {
                i = programLen;
            }
            if (program[i] == 'm') {
                state = ParserState::M;
            }
            ++i;
            break;
        case ParserState::M:
            if (program[i] == 'u') {
                state = ParserState::U;
                ++i;
            } else {
                state = ParserState::NONE;
            }
            break;
        case ParserState::U:
            if (program[i] == 'l') {
                state = ParserState::L;
                ++i;
            } else {
                state = ParserState::NONE;
            }
            break;
        case ParserState::L:
            if (program[i] == '(') {
                state = ParserState::LP;
                ++i;
            } else {
                state = ParserState::NONE;
            }
            break;
        case ParserState::LP:
            if ('0' <= program[i] && program[i] <= '9') {
                state = ParserState::NUM1;
                num1 = program[i] - '0';
                ++i;
            } else {
                state = ParserState::NONE;
            }
            break;
        case ParserState::NUM1:
            if ('0' <= program[i] && program[i] <= '9') {
                num1 = 10 * num1;
                num1 += program[i] - '0';
                ++i;
            } else if (program[i] == ',') {
                state = ParserState::COMMA;
                ++i;
            } else {
                state = ParserState::NONE;
            }
            break;
        case ParserState::COMMA:
            if ('0' <= program[i] && program[i] <= '9') {
                state = ParserState::NUM2;
                num2 = program[i] - '0';
                ++i;
            } else {
                state = ParserState::NONE;
            }
            break;
        case ParserState::NUM2:
            if ('0' <= program[i] && program[i] <= '9') {
                num2 = 10 * num2;
                num2 += program[i] - '0';
                ++i;
            } else if (program[i] == ')') {
                state = ParserState::RP;
            } else {
                state = ParserState::NONE;
            }
            break;
        case ParserState::RP:
            ret += num1 * num2;
            ++i;
            state = ParserState::NONE;
            break;
        }
    }
    return ret;
}

__global__ void do_mul(const char *program, int *programLen, int *result) {
    __shared__ int cache[threadsPerBlock];

    // divide window width by 2 to get overlapped windows
    int programIdx = (WINDOW_W / 2) * (threadIdx.x + blockIdx.x * blockDim.x);
    int cacheIdx = threadIdx.x;

    int partial = 0;

    while (programIdx < programLen[0]) {
        partial += check_program_window(&program[programIdx], std::min(programLen[0] - programIdx + 1, WINDOW_W));
        programIdx += (WINDOW_W / 2) * blockDim.x * gridDim.x;
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
        result[blockIdx.x] = cache[0];
    }
}

int main() {
    std::string program;

    std::string data;
    while (std::cin >> data) {
        program += data;
    }
    int *partialResult = new int[blocksPerGrid];
    int programLen = program.length();

    char *dev_program;
    int *dev_programLen;
    int *dev_partialResult;

    CUDA_CHECK(cudaMalloc((void **)&dev_program, programLen * sizeof(char)));
    CUDA_CHECK(cudaMalloc((void **)&dev_programLen, sizeof(size_t)));
    CUDA_CHECK(cudaMalloc((void **)&dev_partialResult, blocksPerGrid * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dev_program, program.data(), programLen * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_programLen, &programLen, sizeof(int), cudaMemcpyHostToDevice));

    do_mul<<<blocksPerGrid, threadsPerBlock>>>(dev_program, dev_programLen, dev_partialResult);

    CUDA_CHECK(cudaMemcpy(partialResult, dev_partialResult, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));

    int result = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        result += partialResult[i];
    }

    std::cout << "Sum of products: " << result << std::endl;

    cudaFree(dev_program);
    cudaFree(dev_partialResult);

    delete[] partialResult;
}
