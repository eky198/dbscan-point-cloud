/**
 * Parallel DClust+ via CUDA and OpenMP
 * Ethan Ky (etky), Nicholas Beach (nbeach)
 */

#include "dbscan.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>
#include <bits/stdc++.h>

#include <omp.h>
// #include "cycle_timer.h"

#define BLOCKS 256
#define THREADS_PER_BLOCK 512
#define SEED_LIST_SIZE 1024

#define UNPROC -2
#define NOISE -1

struct Parameters {
    float4* point_cloud;
    int num_pts;

    float epsilon;
    int min_pts;
};

__constant__ Parameters device_params;

__device__ inline float distance(float4 point1, float4 point2) {
    return sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2) + pow(point1.z - point2.z, 2) + pow(point1.w - point2.w, 2));
}

__device__ void process_point(int point_idx, int* cluster_list, int* seed_lists, int* seed_list_sizes, int* collision_matrix, int* correction_matrix) {
    int cluster_idx = blockIdx.x;
    int old_cluster_idx = atomicCAS(&cluster_list[point_idx], UNPROC, cluster_idx);
    
    if (old_cluster_idx == UNPROC && seed_list_sizes[cluster_idx] < SEED_LIST_SIZE) {
        int seed_list_size = atomicAdd(&seed_list_sizes[cluster_idx], 1);
        seed_lists[cluster_idx * SEED_LIST_SIZE + seed_list_size] = point_idx;
    }

    if (old_cluster_idx < gridDim.x && old_cluster_idx != cluster_idx && old_cluster_idx != NOISE) {
        if (old_cluster_idx < cluster_idx) {
            collision_matrix[old_cluster_idx * BLOCKS + cluster_idx] = 1;
        }
    }

    if (old_cluster_idx >= gridDim.x) {
        for (int i = 0; i < device_params.num_pts; i++) {
            int changed_cluster_idx = atomicCAS(&correction_matrix[cluster_idx * device_params.num_pts + i], UNPROC, old_cluster_idx);
            if (changed_cluster_idx == UNPROC || changed_cluster_idx == old_cluster_idx) {
                break;
            }
        }
    }

    if (old_cluster_idx == NOISE) {
        atomicCAS(&cluster_list[point_idx], NOISE, cluster_idx);
    }
}

__global__ void expansion_kernel(int* cluster_list, int* seed_lists, int* seed_list_sizes, int* collision_matrix, int* correction_matrix) {
    int cluster_idx = blockIdx.x;
    __shared__ int quarantine[SEED_LIST_SIZE];
    __shared__ int neighbor_count;
    __shared__ int seed_point_idx;
    __shared__ float4 seed_point;

    while (seed_list_sizes[cluster_idx] > 0) {
        if (threadIdx.x == 0) {
            // printf("The device params are: epsilon = %f, min_pts = %d\n", device_params.epsilon, device_params.min_pts);
            int seed_list_size = atomicSub(&seed_list_sizes[cluster_idx], 1);
            seed_point_idx = seed_lists[cluster_idx * SEED_LIST_SIZE + seed_list_size - 1];
            seed_point = device_params.point_cloud[seed_point_idx];
        }
        __syncthreads();

        for (int point_idx = threadIdx.x; point_idx < device_params.num_pts; point_idx += THREADS_PER_BLOCK) {
            float4 point = device_params.point_cloud[point_idx];
            if (distance(point, seed_point) <= device_params.epsilon) {
                atomicAdd(&neighbor_count, 1);
                if (neighbor_count < device_params.min_pts) {
                    quarantine[neighbor_count] = point_idx;
                }
                else {
                    process_point(point_idx, cluster_list, seed_lists, seed_list_sizes, collision_matrix, correction_matrix);
                }
            }
        }
        __syncthreads();

        if (neighbor_count >= device_params.min_pts) {
            cluster_list[seed_point_idx] = cluster_idx;
            int quarantine_idx = threadIdx.x;
            if (quarantine_idx < device_params.min_pts) {
                process_point(quarantine[quarantine_idx], cluster_list, seed_lists, seed_list_sizes, collision_matrix, correction_matrix);
            }
        }
        else {
            cluster_list[seed_point_idx] = NOISE;
        }
        __syncthreads();
    }
}

__global__ void fill_seed_list_kernel(int* cluster_list, int* seed_lists, int* seed_list_sizes, bool* result) {
    __shared__ int next_cluster_idx;

    for (int point_idx = threadIdx.x; point_idx < device_params.num_pts; point_idx += THREADS_PER_BLOCK) {
        if (cluster_list[point_idx] == UNPROC) {
            *result = false;
            if (next_cluster_idx < BLOCKS) {
                int old_cluster_idx = atomicAdd(&next_cluster_idx, 1);
                if (old_cluster_idx < BLOCKS) {
                    seed_lists[old_cluster_idx * SEED_LIST_SIZE] = point_idx;
                    seed_list_sizes[old_cluster_idx] = 1;
                }
            }
        }
    }
}

__global__ void init_matrix_kernel(int* collision_matrix, int* correction_matrix, int num_pts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BLOCKS * BLOCKS) {
        collision_matrix[idx] = UNPROC;
    }
    if (idx < BLOCKS) {
        correction_matrix[idx] = UNPROC;
    }
}

__global__ void merge_clusters_kernel(int* cluster_list, int* collision_matrix, int* correction_matrix, int* next_cluster) {
    int cluster_idx = blockIdx.x;
    int new_cluster_label = *next_cluster + cluster_idx;

    for (int point_idx = threadIdx.x; point_idx < device_params.num_pts; point_idx += THREADS_PER_BLOCK) {
        int point_cluster_idx = cluster_list[point_idx];
        if (point_cluster_idx != UNPROC && point_cluster_idx != NOISE && point_cluster_idx < BLOCKS) {
            // Check for any collision
            if (collision_matrix[point_cluster_idx * BLOCKS + cluster_idx] != UNPROC) {
                cluster_list[point_idx] = cluster_idx;
            }
        }
    }
    __syncthreads();

    for (int point_idx = threadIdx.x; point_idx < device_params.num_pts; point_idx += THREADS_PER_BLOCK) {
        int point_cluster_idx = cluster_list[point_idx];

        // Check if the point already has intermediate cluster, then merge 
        if (point_cluster_idx >= gridDim.x) {
            for (int i = 0; i < device_params.num_pts; i++) {
                int new_cluster_idx = correction_matrix[cluster_idx * device_params.num_pts + i];
                if (new_cluster_idx == UNPROC) {
                    break;
                }

                if (new_cluster_idx == point_cluster_idx) {
                    cluster_list[point_idx] = correction_matrix[cluster_idx * device_params.num_pts];
                    break;
                }
            }
        }

        // Otherwise, check if the point is processed and has same chain idea as current block, then assign a new label
        if (point_cluster_idx != UNPROC && point_cluster_idx != NOISE && point_cluster_idx == cluster_idx) {
            cluster_list[point_idx] = new_cluster_label;
        }
    }
    __syncthreads();

    if (cluster_idx == 0) {
        atomicAdd(next_cluster, BLOCKS);
    }
}

void finalize_clusters(int* cluster_list, int num_pts) {
    int i;
    #pragma omp parallel for default(shared) private(i) schedule(static)
        for (i = 0; i < num_pts; i++) {
            if (cluster_list[i] != NOISE) {
                cluster_list[i] -= BLOCKS;
            }
        } 
}

void dbscan(std::vector<float4>& points, int* cluster_list, double epsilon, int min_pts) {
    int num_pts = points.size();

    // put point cloud in device memory
    float4* device_point_cloud;
    cudaMalloc(&device_point_cloud, sizeof(float4) * num_pts);
    cudaMemcpy(device_point_cloud, points.data(), sizeof(float4) * num_pts, cudaMemcpyHostToDevice);
    printf("Allocated point cloud in device memory\n");

    // put parameters in constant memory
    Parameters params{device_point_cloud, num_pts, epsilon, min_pts};
    cudaMemcpyToSymbol(device_params, &params, sizeof(Parameters));
    printf("Allocated parameters in device memory\n");

    // cluster lists
    int* device_cluster_list;
    cudaMalloc(&device_cluster_list, sizeof(int) * num_pts);
    cudaMemcpy(device_cluster_list, cluster_list, sizeof(int) * num_pts, cudaMemcpyHostToDevice);
    printf("Allocated cluster list in device memory\n");

    // collision and correction matrices
    int* device_collision_matrix;
    cudaMalloc(&device_collision_matrix, sizeof(int) * BLOCKS * BLOCKS);
    int* device_correction_matrix;
    cudaMalloc(&device_correction_matrix, sizeof(int) * BLOCKS * num_pts);
    // cudaMalloc(&device_correction_matrix, sizeof(int) * BLOCKS);
    int* collision_matrix = new int[BLOCKS * BLOCKS];
    int* correction_matrix = new int[BLOCKS * num_pts];
    // int* correction_matrix = new int[BLOCKS];
    printf("Allocated collision and correction matrices in device memory\n");

    // seed list variables
    int* device_seed_lists;
    cudaMalloc(&device_seed_lists, sizeof(int) * BLOCKS * SEED_LIST_SIZE);
    cudaMemset(device_seed_lists, 0, sizeof(int) * BLOCKS * SEED_LIST_SIZE);
    int* device_seed_list_sizes;
    cudaMalloc(&device_seed_list_sizes, sizeof(int) * BLOCKS);
    cudaMemset(device_seed_list_sizes, 0, sizeof(int) * BLOCKS);
    bool* device_result;
    cudaMalloc(&device_result, sizeof(bool));
    printf("Allocated seed lists in device memory\n");

    // start maintaining next cluster label
    // int next_cluster = BLOCKS;
    int* device_next_cluster;
    cudaMalloc(&device_next_cluster, sizeof(int));
    cudaMemset(device_next_cluster, BLOCKS, sizeof(int));

    int iter = 0;
    fill_seed_list_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(device_cluster_list, device_seed_lists, device_seed_list_sizes, device_result);
    while (true) {
        printf("Running iteration %d\n", iter);
        init_matrix_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(device_collision_matrix, device_correction_matrix, num_pts);
        cudaDeviceSynchronize();
        expansion_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(device_cluster_list, device_seed_lists, device_seed_list_sizes, device_collision_matrix, device_correction_matrix);
        cudaDeviceSynchronize();

        merge_clusters_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(device_cluster_list, device_collision_matrix, device_correction_matrix, device_next_cluster);
        cudaDeviceSynchronize();

        bool result = true;
        cudaMemcpy(device_result, &result, sizeof(bool), cudaMemcpyHostToDevice);
        fill_seed_list_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(device_cluster_list, device_seed_lists, device_seed_list_sizes, device_result);
        cudaDeviceSynchronize();
        cudaMemcpy(&result, device_result, sizeof(bool), cudaMemcpyDeviceToHost);

        iter++;
        if (result) {
            break;
        }
    }

    cudaMemcpy(cluster_list, device_cluster_list, sizeof(int) * num_pts, cudaMemcpyDeviceToHost);
    finalize_clusters(cluster_list, num_pts);

    // free host memory
    delete [] collision_matrix;
    delete [] correction_matrix;

    // free cuda memory
    cudaFree(device_point_cloud);
    cudaFree(device_cluster_list);
    cudaFree(device_collision_matrix);
    cudaFree(device_correction_matrix);
    cudaFree(device_seed_lists);
    cudaFree(device_seed_list_sizes);
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

    cuda_setup();

    std::vector<float4> points;

    std::string line;
    while (std::getline(fin, line)) {
        std::istringstream sin(line);
        float4 point;
        for (int i = 0; i < DIMENSIONALITY; i++) {
            sin >> point.x >> point.y >> point.z >> point.w;
        }
        points.push_back(point);
    }
    fin.close();
    size_t num_pts = points.size();
    std::cout << "Number of points in input: " << num_pts << '\n';

    omp_set_num_threads(num_threads);

    /* Initialize additional data structures */
    int* cluster_list = new int[num_pts];
    size_t i;
    #pragma omp parallel for default(shared) private(i) schedule(static)
        for (i = 0; i < num_pts; i++) {
            cluster_list[i] = UNPROC;
        }

    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

    /* Perform all computation here */
    const auto compute_start = std::chrono::steady_clock::now();

    dbscan(points, cluster_list, epsilon, min_pts);

    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << compute_time << '\n';

    const double total_time = init_time + compute_time;
    std::cout << "Total time (sec): " << total_time << '\n';

    std::vector<Point> point_cloud(num_pts);
    #pragma omp parallel for default(shared) private(i) schedule(static)
        for (i = 0; i < num_pts; i++) {
            Point point;
            point.cluster = cluster_list[i];
            point_cloud[i] = point;
        }

    write_output(point_cloud, num_threads, epsilon, min_pts, input_filename);

    delete [] cluster_list;
}