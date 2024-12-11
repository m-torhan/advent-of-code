#include "utils.cuh"
#include <iostream>
#include <sstream>
#include <vector>

constexpr int N = 1024;
constexpr int threadsPerBlock = 256;
constexpr int blocksPerGrid = std::min(32, 1 + (N - 1) / threadsPerBlock);

__device__ bool safe_condition(int delta, int sign) {
    return (
        /* any two adjacent levels differ by at least one and at most three */
        (abs(delta) >= 1 && abs(delta) <= 3) &&
        /* the levels are either all increasing or all decreasing */
        ((delta > 0 && sign > 0) || (delta < 0 && sign < 0)));
}

__device__ bool is_report_safe(int *report, int reportLen) {
    bool ret = true;

    int *deltas = (int *)malloc((reportLen - 1) * sizeof(int));
    int errors = 0;
    int errorIdx1 = -1;
    int errorIdx2 = -1;
    int sign = 0;

    // compute deltas between values
    for (int i = 0; i < reportLen - 1; ++i) {
        deltas[i] = report[i + 1] - report[i];
        sign += (deltas[i] > 0) - (deltas[i] < 0);
    }

    if (sign == 0) {
        sign = 1;
    }

    // check if safe
    for (int i = 0; i < reportLen - 1; ++i) {
        if (!safe_condition(deltas[i], sign)) {
            if (errors == 0) {
                errorIdx1 = i;
            } else if (errors == 1) {
                errorIdx2 = i;
            }
            ++errors;
            ret = false;
        }
    }

    // try fix the report
    if (errors == 1) {
        if (errorIdx1 == 0 || errorIdx1 == reportLen - 2) {
            // error at the beggining or end of report
            ret = true;
        }
        if (errorIdx1 > 0) {
            // try removing first element
            if (safe_condition(deltas[errorIdx1 - 1] + deltas[errorIdx1], sign)) {
                ret = true;
            }
        }
        if (errorIdx1 < reportLen - 2) {
            // try removing second element
            if (safe_condition(deltas[errorIdx1] + deltas[errorIdx1 + 1], sign)) {
                ret = true;
            }
        }
    } else if (errors == 2 && errorIdx2 == (errorIdx1 + 1)) {
        if (safe_condition(deltas[errorIdx1] + deltas[errorIdx1 + 1], sign)) {
            ret = true;
        }
    }

    free(deltas);

    return ret;
}

__global__ void are_reports_safe(int *report, int *reportLen, int *safe, int maxReportLen, size_t size) {
    __shared__ int cache[threadsPerBlock];

    int reportIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    int partial = 0;
    while (reportIdx < size) {
        partial += is_report_safe(&report[reportIdx * maxReportLen], reportLen[reportIdx]);
        reportIdx += blockDim.x * gridDim.x;
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
        safe[blockIdx.x] = cache[0];
    }
}

int main() {
    std::vector<std::vector<int>> reportsVec;

    size_t maxReportLen = 0;

    for (std::string line; std::getline(std::cin, line);) {
        std::stringstream sline(line);
        reportsVec.push_back({});

        int num;
        while (sline >> num) {
            reportsVec.back().push_back(num);
        }
        maxReportLen = std::max(maxReportLen, reportsVec.back().size());
    }

    const auto reportsNum = reportsVec.size();

    int *reports = new int[maxReportLen * reportsNum];
    int *reportsLen = new int[reportsNum];

    for (int i = 0; i < reportsNum; ++i) {
        reportsLen[i] = reportsVec[i].size();
        memcpy(&reports[i * maxReportLen], reportsVec[i].data(), reportsVec[i].size() * sizeof(int));
    }

    auto *partialSum = new int[blocksPerGrid];
    int *dev_reports, *dev_reportsLen, *dev_partialSum;

    CUDA_CHECK(cudaMalloc((void **)&dev_reports, maxReportLen * reportsNum * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_reportsLen, reportsNum * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_partialSum, blocksPerGrid * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dev_reports, reports, maxReportLen * reportsNum * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_reportsLen, reportsLen, reportsNum * sizeof(int), cudaMemcpyHostToDevice));

    are_reports_safe<<<blocksPerGrid, threadsPerBlock>>>(dev_reports, dev_reportsLen, dev_partialSum, maxReportLen,
                                                         reportsNum);

    CUDA_CHECK(cudaMemcpy(partialSum, dev_partialSum, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));

    int sum = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        sum += partialSum[i];
    }

    std::cout << "Safe reports count: " << sum << std::endl;

    cudaFree(dev_reports);
    cudaFree(dev_reportsLen);
    cudaFree(dev_partialSum);

    delete[] reports;
    delete[] reportsLen;
    delete[] partialSum;
}
