/**
 * Parallel Disjoint-Set DBSCAN via OpenMP
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

#include <unistd.h>
#include <omp.h>

struct dbscan_mngr_t {
    std::vector<int> times;
    std::vector<int> points_to_threads;
    std::vector<int> threads_to_points;
} * dbscan_mngr;

void dbscan(PointCloud& point_cloud, int num_threads, double epsilon, int min_pts) {
    DisjointSetInt disjoint_sets(point_cloud.size());

    const size_t block_size = (point_cloud.size() + num_threads - 1) / num_threads;
    
    omp_set_num_threads(num_threads);

    // Set up manager to keep track of partitions and timing
    auto dbscan_mngr = new dbscan_mngr_t;

    std::vector<std::vector<std::pair<int, int>>> pairs(num_threads);
    int t;
    #pragma omp parallel for default(shared) private(t) schedule(static)
        for (t = 0; t < num_threads; t++) {
            pairs[t].reserve(block_size * point_cloud.size());
        }

    // First form clusters within threads
    #pragma omp parallel for default(shared) private(t) schedule(dynamic)
        for (t = 0; t < num_threads; t++) {
            int start_idx = t * block_size;
            int end_idx = std::min(point_cloud.size(), start_idx + block_size);

            for (int i = start_idx; i < end_idx; i++) {
                Point& x = point_cloud[i];
                auto neighbors = point_cloud.get_neighbors(x, epsilon);
                if (neighbors.size() >= min_pts) {
                    x.status = core;
                    for (auto neighbor : neighbors) {
                        int j = neighbor.first;
                        Point& y = point_cloud[j];
                        if (start_idx <= j && j < end_idx) {
                            if (y.status == core) {
                                disjoint_sets.union_set(i, j);
                            }
                            else if (y.status == none) {
                                y.status = border;
                                disjoint_sets.union_set(i, j);
                            }
                        }
                        else {
                            pairs[t].emplace_back(i, j);
                        }
                    }
                }
            }
        }

    if (omp_get_thread_num() == 0) {
        std::cout << "Finished forming in-thread clusters\n";
    }

    // Then merge clusters across threads
    #pragma omp parallel for default(shared) private(t) schedule(dynamic)
        for (t = 0; t < num_threads; t++) {
            printf("Processing set number %d with %ld pairs\n", t, pairs[t].size());
            for (int idx = 0; idx < pairs[t].size(); idx++) {
                int i = pairs[t][idx].first;
                int j = pairs[t][idx].second;
                Point& y = point_cloud[j];

                if (y.status == core) {
                    disjoint_sets.union_set_with_lock(i, j);
                }
                // #pragma omp critical
                    if (y.status == none) {
                        y.status = border;
                        disjoint_sets.union_set_with_lock(i, j);
                    }
            }
        }

    // Then do path compression and label clusters
    size_t i;
    #pragma omp parallel for default(shared) private(i) schedule(dynamic)
        for (i = 0; i < point_cloud.size(); i++) {
            Point& point = point_cloud[i];
            int j = disjoint_sets.find_set(i);
            Point& parent = point_cloud[j];
            if (parent.status != none) {
                #pragma omp critical
                    if (parent.cluster < 0) {
                        parent.cluster = point_cloud.next_cluster++;
                    }
                point.cluster = parent.cluster;
            }
        }

    delete dbscan_mngr;
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