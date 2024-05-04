/**
 * Parallel DClust+ via CUDA
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

void dbscan(point_cloud, num_threads, epsilon, min_pts) {
    
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

    PointCloud point_cloud;

    std::string line;
    while (std::getline(fin, line)) {
        std::istringstream sin(line);
        Point point;
        point.cluster = -1;
        for (int i = 0; i < DIMENSIONALITY; i++) {
            sin >> point.data[i];
        }
        point_cloud.add_point(point);
    }
    fin.close();
    size_t num_pts = point_cloud.size();
    std::cout << "Number of points in input: " << num_pts << '\n';

    /* Initialize additional data structures */

    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

    /* Perform all computation here */
    const auto compute_start = std::chrono::steady_clock::now();

    dbscan(point_cloud, num_threads, epsilon, min_pts);

    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << compute_time << '\n';

    const double total_time = init_time + compute_time;
    std::cout << "Total time (sec): " << total_time << '\n';

    write_output(point_cloud, num_threads, input_filename);
}