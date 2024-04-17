/**
 * General Structures and Functions for Parallel and Sequential DBSCAN
 * Ethan Ky (etky), Nicholas Beach (nbeach)
 */

#ifndef __DBSCAN_H__
#define __DCSCAN_H__

#include <omp.h>
#include <vector>
#include <cmath>
#include <boost/pending/disjoint_sets.hpp>

#include "nanoflann.hpp"

#define VALUES_PER_POINT 4

enum status { none, core, border, noise };

struct Point {
    float data[VALUES_PER_POINT]; 
    status status;
    int cluster;

    float x(void) const { return data[0]; }
    float y(void) const { return data[1]; }
    float z(void) const { return data[2]; }
    float r(void) const { return data[3]; }
};

inline float dist(const Point& pt1, const Point& pt2) {
    return sqrt((pt1.x() - pt2.x()) * (pt1.x() - pt2.x()) + 
                (pt1.y() - pt2.y()) * (pt1.y() - pt2.y()) + 
                (pt1.z() - pt2.z()) * (pt1.z() - pt2.z()) + 
                (pt1.r() - pt2.r()) * (pt1.r() - pt2.r())); 
}

// TODO: Construction of KD-Tree
// TODO: Wrapper functions for radius search and/or nearest neighbor search
struct PointCloud {
    std::vector<Point> points;
    int next_cluster = 0;

    PointCloud(int num_pts) {
        points.resize(num_pts);
    }
    
    int size(void) const {
        return points.size();
    }

    Point& operator[](int i) {
        return points[i];
    }

    Point operator[](int i) const {
        return points[i];
    }
};

/* Sequential DBSCAN */

/* Disjoint-Set DBSCAN via OpenMP */

/* DBSCAN via CUDA */

void write_output(const PointCloud& point_cloud, int num_threads, std::string input_filename) {
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
    out_clusters << num_points << '\n';
    for (int i = 0; i < num_points; i++) {
        out_clusters << point_cloud[i].cluster << ' ';
    }
    out_clusters << '\n';
    out_clusters.close();
}





