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
#include <map>

#define DIMENSIONALITY 4

/* Point Cloud Structures */

enum status { none, core, border, noise };

typedef struct Point {
    double data[DIMENSIONALITY]; 
    status status = none;
    int cluster = -1;

    double x(void) const { return data[0]; }
    double y(void) const { return data[1]; }
    double z(void) const { return data[2]; }
    double r(void) const { return data[3]; }
} Point;

struct PointCloud {
    std::vector<Point> points;
    int next_cluster = 0;

    typedef typename nanoflann::metric_L2::template traits<double, PointCloud>::distance_t metric_t;
    nanoflann::KDTreeSingleIndexDynamicAdaptor<metric_t, PointCloud, DIMENSIONALITY, int>* index;

    PointCloud(int num_pts, int max_leaf_size = 10) {
        points.resize(num_pts);
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
        // TODO
    }

    std::vector<nanoflann::ResultItem<int, double>> get_neighbors(int idx, double radius) {
        return get_neighbors(points[idx], radius);
    }

    std::vector<nanoflann::ResultItem<int, double>> get_neighbors(const Point point, double radius) {
        std::vector<nanoflann::ResultItem<int, double>> result_vec;
        nanoflann::RadiusResultSet<double, int> result(radius, result_vec);
        result.init();
        index->findNeighbors(result, point.data);
        return result_vec;
    }
};

/* Disjoint Sets */
typedef std::map<Point, std::size_t> rank_t;
typedef std::map<Point, Point> parent_t;
typedef boost::disjoint_sets< boost::associative_property_map<rank_t>,  boost::associative_property_map<parent_t>> disjointset_t;

disjointset_t make_disjoint_sets(std::vector<Point> points) {
    rank_t rank_map;
    parent_t parent_map;
    boost::associative_property_map<rank_t>   rank_pmap(rank_map);
    boost::associative_property_map<parent_t> parent_pmap(parent_map);

    disjointset_t disjoint_sets(rank_map, parent_map);
    for (Point& point : points) {
        disjoint_sets.make_set(point);
    }

    return disjoint_sets;
}

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





