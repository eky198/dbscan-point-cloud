/**
 * General Structures and Functions for Parallel and Sequential DBSCAN
 * Ethan Ky (etky), Nicholas Beach (nbeach)
 */

#ifndef __DBSCAN_H__
#define __DCSCAN_H__

#include <omp.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <unordered_map>

#include "nanoflann.hpp"
#include <boost/functional/hash.hpp>

#define DIMENSIONALITY 4
#define NUM_PRINT_POINTS 5

/* Point Cloud Structures */

enum point_status { none, core, border, noise };

struct Point {
    double data[DIMENSIONALITY]; 
    point_status status = none;
    int cluster = -1;

    double x(void) const { return data[0]; }
    double y(void) const { return data[1]; }
    double z(void) const { return data[2]; }
    double r(void) const { return data[3]; }

    struct Hash {
        size_t operator()(const Point& point) const {
            size_t seed = 0;
            for (int idx = 0; idx < DIMENSIONALITY; idx++) {
                boost::hash_combine(seed, point.data[idx]);
            }
            return seed;
        }
    };

    int operator==(const Point& other) const {
        for (int idx = 0; idx < DIMENSIONALITY; idx++) {
            if (data[idx] != other.data[idx]) {
                return false;
            }
        }
        return true;
    }

    int operator!=(const Point& other) const {
        for (int idx = 0; idx < DIMENSIONALITY; idx++) {
            if (data[idx] != other.data[idx]) {
                return true;
            }
        }
        return false;
    }
};

struct PointCloud {
    std::vector<Point> points;
    std::atomic<int> next_cluster{0};

    typedef nanoflann::L2_Simple_Adaptor<double, PointCloud> metric_t;
    nanoflann::KDTreeSingleIndexDynamicAdaptor<metric_t, PointCloud, DIMENSIONALITY, int>* index;

    PointCloud(int max_leaf_size = 10) {
        index = new nanoflann::KDTreeSingleIndexDynamicAdaptor<metric_t, PointCloud, DIMENSIONALITY, int>(DIMENSIONALITY, *this, nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size));
    }

    ~PointCloud() {
        delete index;
    }
    
    /* DatasetAdaptor interface for KDTreeSingleIndexDynamicAdaptor */

    inline size_t kdtree_get_point_count() const {
        return points.size();
    }

    inline double kdtree_get_pt(const size_t idx, int dim) const {
        return points[idx].data[dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX &bb) const
    {
        return false;
    }

    /* Member functions */
    
    size_t size() const {
        return points.size();
    }

    Point& operator[](int idx) {
        return points[idx];
    }

    Point operator[](int idx) const {
        return points[idx];
    }

    void add_point(Point& point) {
        points.push_back(point);
        index->addPoints(size() - 1, size() - 1);
    }

    void remove_point(int idx) {
        // TODO: Implement this function
    }

    std::vector<nanoflann::ResultItem<int, double>> get_neighbors(int idx, double radius) {
        return get_neighbors(points[idx], radius);
    }

    std::vector<nanoflann::ResultItem<int, double>> get_neighbors(const Point point, double radius) {
        std::vector<nanoflann::ResultItem<int, double>> result_vec;
        nanoflann::RadiusResultSet<double, int> result(pow(radius, 2), result_vec);
        result.init();
        index->findNeighbors(result, point.data);
        return result.m_indices_dists;
    }
};

/* Disjoint Set Structures */
struct DisjointSetInt {
    std::vector<int> parent;
    std::vector<omp_lock_t> lock;

    DisjointSetInt(int num_pts) {
        parent.resize(num_pts);
        lock.resize(num_pts);
        for (int i = 0; i < num_pts; i++) {
            parent[i] = i;
            omp_init_lock(&lock[i]);
        }
    }

    ~DisjointSetInt() {
        for (size_t i = 0; i < lock.size(); i++) {
            omp_destroy_lock(&lock[i]);
        }
    }

    int find_set(int i) {
        while (i != parent[i]) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    }

    void union_set(int i, int j) {
        while (parent[i] != parent[j]) {
            if (parent[i] < parent[j]) {
                if (i == parent[i]) {
                    parent[i] = parent[j];
                }
                i = parent[i];
            }
            else {
                if (j == parent[j]) {
                    parent[j] = parent[i];
                }
                j = parent[j];
            }
        }
    }

    void union_set_with_lock(int i, int j) {
        while (parent[i] != parent[j]) {
            if (parent[i] < parent[j]) {
                if (i == parent[i]) {
                    omp_set_lock(&lock[i]);
                    if (i == parent[i]) {
                        parent[i] = parent[j];
                    }
                    omp_unset_lock(&lock[i]);
                }
                i = parent[i];
            }
            else {
                if (j == parent[j]) {
                    omp_set_lock(&lock[j]);
                    if (j == parent[j]) {
                        parent[j] = parent[i];
                    }
                    omp_unset_lock(&lock[j]);
                }
                j = parent[j];
            }
        }
    }
};

void write_output(const PointCloud& point_cloud, int num_threads, double epsilon, int min_pts, std::string input_filename) {
    if (input_filename.size() >= 4 && input_filename.substr(input_filename.size() - 4) == ".txt") {
        input_filename.resize(input_filename.size() - 4);
    }

    const std::string clusters_filename = input_filename + "_clusters_" + std::to_string(num_threads) + ".txt";

    std::ofstream out_clusters(clusters_filename, std::fstream::out);
    if (!out_clusters) {
        std::cerr << "Unable to open file: " << clusters_filename << '\n';
        exit(EXIT_FAILURE);
    }

    int num_points = point_cloud.size();
    std::cout << "Writing " << num_points << " points\n";
    out_clusters << num_points << '\n';
    out_clusters << epsilon << ' ' << min_pts << '\n';
    for (int i = 0; i < num_points; i++) {
        out_clusters << point_cloud[i].cluster << '\n';
    }
    out_clusters << '\n';
    out_clusters.close();
}

#endif





