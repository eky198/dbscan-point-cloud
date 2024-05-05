/**
 * Parallel DClust+ via CUDA
 * Ethan Ky (etky), Nicholas Beach (nbeach)
 */

#include "dbscan.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math_functions.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>

#define BLOCKS 256
#define THREADS_PER_BLOCK 512

__device__ inline float
distance(float4 point1, float4 point2) {
    return sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2) + pow(point1.z - point2.z, 2) + pow(point1.w - point2.w, 2))
}

__global__ void index_kernel(float4* point_cloud, float epsilon, int min_pts, int r) {

}

__global__ void expansion_kernel(float4* point_cloud, float epsilon, int min_pts, int r) {

}

bool fill_seed_list(float4* point_cloud) {

}

void merge_clusters() {

}

void dbscan(std::vector<float4> points, int num_threads, double epsilon, int min_pts) {
    float4* device_point_cloud;
    int rounded_length = nextPow2(point_cloud.size());
    cudaMalloc((void **)&device_point_cloud, sizeof(int) * rounded_length);
    cudaMemcpy(device_point_cloud, points.data(), points.size() * sizeof(float4), cudaMemcpyHostToDevice);

    index_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>()
}

void cuda_setup() {
    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("GeForce RTX 2080") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA RTX 2080.\n");
        printf("---------------------------------------------------------\n");
    }
}

int main(int argc, char *argv[]) {

    /* Initialize parameters and read point cloud data */

    const auto init_start = std::chrono::steady_clock::now();
    std::string input_filename;
    int num_threads = 1;
    double epsilon = 5;
    int min_pts = 10;

    int opt;
    while ((opt = getopt(argc, argv, "f:n:e:p:")) != -1) {
        switch (opt) {
        case 'f':
            input_filename = optarg;
            break;
        case 'n':
            num_threads = atoi(optarg);
            break;
        case 'e':
            epsilon = atof(optarg);
            break;
        case 'p':
            min_pts = atoi(optarg);
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " -f input_filename -n num_threads [-e epsilon] [-p min_pts]\n";
            exit(EXIT_FAILURE);
        }
    }

    if (input_filename.empty() || num_threads <= 0 || epsilon <= 0 || min_pts <= 0) {
        std::cerr << "Usage: " << argv[0] << " -f input_filename -n num_threads [-e epsilon] [-p min_pts]\n";
        exit(EXIT_FAILURE);
    }

    std::cout << "Data file: " << input_filename << '\n';
    std::cout << "Number of threads (num_threads): " << num_threads << '\n';
    std::cout << "Distance threshold (epsilon): " << epsilon << '\n';
    std::cout << "Minimum number of points to form a cluster (min_pts): " << min_pts << '\n';

    std::ifstream fin(input_filename);
    if (!fin) {
        std::cerr << "Unable to open file: " << input_filename << ".\n";
        exit(EXIT_FAILURE);
    }

    std::vector<float4> points;

    std::string line;
    while (std::getline(fin, line)) {
        std::istringstream sin(line);
        float4 point;
        for (int i = 0; i < DIMENSIONALITY; i++) {
            sin >> point.x >> point.y >> point.z >> point.w;
        }
        points.push(point);
    }
    fin.close();
    size_t num_pts = points.size();
    std::cout << "Number of points in input: " << num_pts << '\n';

    /* Initialize additional data structures */

    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

    /* Perform all computation here */
    const auto compute_start = std::chrono::steady_clock::now();

    dbscan(points, num_threads, epsilon, min_pts);

    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << compute_time << '\n';

    const double total_time = init_time + compute_time;
    std::cout << "Total time (sec): " << total_time << '\n';

    // TODO: Create point cloud object here

    write_output(point_cloud, num_threads, epsilon, min_pts, input_filename);
}