/**
 * Sequential DBSCAN
 * Ethan Ky (etky), Nicholas Beach (nbeach)
 */

#include "dbscan.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>

void dbscan(PointCloud& point_cloud, double epsilon, int min_pts) {

}

int main(int argc, char *argv[]) {

    /* Initialize parameters and read point cloud data */

    const auto init_start = std::chrono::steady_clock::now();
    std::string input_filename;
    double epsilon = 0;
    int min_pts = 0;

    int opt;
    while ((opt = getopt(argc, argv, "f:e:p:")) != -1) {
        switch (opt) {
        case 'f':
            input_filename = optarg;
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

    if (input_filename.empty() || epsilon <= 0 || min_pts <= 0) {
        std::cerr << "Usage: " << argv[0] << " -f input_filename -n num_threads [-e epsilon] [-p min_pts]\n";
        exit(EXIT_FAILURE);
    }

    std::cout << "Distance threshold (epsilon): " << epsilon << "\n";
    std::cout << "Minimum number of points to form a cluster (min_pts): " << min_pts << "\n";

    std::FILE *p_file = fopen(input_filename.c_str(), "rb");
    fseek(p_file, 0, SEEK_END);
    long file_size = ftell(p_file);
    rewind(p_file);
    float *raw_data = new float[file_size];
    fread(raw_data, sizeof(float), file_size / sizeof(float), p_file);
    long num_pts = file_size / 4 / sizeof(float);
    float **pc_data = new float*[num_pts];
    PointCloud point_cloud(num_pts);
    for (int i = 0; i < num_pts; i++) {
        point_cloud[i].cluster = i;
        for (int j = 0; j < 4; j++) {
            point_cloud[i].data[j] = raw_data[4 * i + j];
        }
    }
    std::cout << "Number of points in input: " << num_pts << "\n";

    /* Initialize additional data structures */

    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

    /* Perform all computation here */
    dbscan(point_cloud, epsilon, min_pts);

    const auto compute_start = std::chrono::steady_clock::now();

    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << compute_time << '\n';

    const double total_time = init_time + compute_time;
    std::cout << "Total time (sec): " << total_time << '\n';
}