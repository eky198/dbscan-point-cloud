/**
 * Sequential Disjoint-Set DBSCAN
 * Ethan Ky (etky), Nicholas Beach (nbeach)
 */

#include "dbscan.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>

void dbscan(PointCloud& point_cloud, double epsilon, int min_pts) {
    DisjointSetInt disjoint_sets(point_cloud.size());

    // First form clusters within threads
    for (size_t i = 0; i < point_cloud.size(); i++) {
        Point& x = point_cloud[i];
        auto neighbors = point_cloud.get_neighbors(x, epsilon);
        if (neighbors.size() >= min_pts) {
            x.status = core;
            for (auto neighbor : neighbors) {
                int j = neighbor.first;
                Point& y = point_cloud[j];
                if (y.status == core) {
                    disjoint_sets.union_set(i, j);
                }
                else if (y.status == none) {
                    y.status = border;
                    disjoint_sets.union_set(i, j);
                }
            }
        }
    }

    // Then do path compression and label clusters
    for (size_t i = 0; i < point_cloud.size(); i++) {
        Point& point = point_cloud[i];
        Point& parent = point_cloud[disjoint_sets.find_set(i)];
        if (parent.status != none) {
            if (parent.cluster < 0) {
                parent.cluster = point_cloud.next_cluster++;
            }
            point.cluster = parent.cluster;
        }
    }
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

    /* Debugging */
    std::cout << "Here are the first " << NUM_PRINT_POINTS << " points: \n";
    for (int i = 0; i < 5; i++) {
        Point point = point_cloud[i];
        std::cout << "x: " << point.data[0] << ", y: " << point.data[1] << ", z: " << point.data[2] << ", r: " << point.data[3] << '\n';
    }

    /* Perform all computation here */
    const auto compute_start = std::chrono::steady_clock::now();

    dbscan(point_cloud, epsilon, min_pts);

    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << compute_time << '\n';

    const double total_time = init_time + compute_time;
    std::cout << "Total time (sec): " << total_time << '\n';

    write_output(point_cloud, 1, epsilon, min_pts, input_filename);
}