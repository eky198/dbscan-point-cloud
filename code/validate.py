import logging
import sys
import numpy as np
from sklearn.cluster import DBSCAN

FORMAT = '%(asctime)-15s - %(levelname)s - %(module)10s:%(lineno)-5d - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)
LOG = logging.getLogger(__name__)

help_message = '''
usage: validate.py [-h] [-d point cloud raw data file] [-c clusters file]

Validate the wire routing output and cost matrix

optional arguments:
  -h, --help            Show this help message and exit
  -v, --verbose         Show all mismatched labels
'''

verbose = False
MISMATCH_REPORT_CNT = 6
mismatch_cnt = 0


def dbscan(point_cloud, epsilon, min_pts):
    clusters = DBSCAN(eps=epsilon, min_samples=min_pts, algorithm='kd_tree', n_jobs=-1).fit(point_cloud)
    return clusters.labels_


def parse_args():
    global verbose
    args = sys.argv
    if '-h' in args or '--help' in args:
        print (help_message)
        sys.exit(1)
    if '-d' not in args or '-c' not in args:
        print ('help_message')
        sys.exit(1)
    if '-v' in args or '--verbose' in args:
        verbose = True
    parsed = {}
    parsed['data'] = args[args.index('-d') + 1]
    parsed['clusters'] = args[args.index('-c') + 1]
    return parsed


def validate(args):
    global mismatch_cnt

    # load point cloud from text file
    point_cloud = np.loadtxt(args['data'])
    point_cloud = np.reshape(point_cloud, (-1, 4))
    print(point_cloud)
    params = np.loadtxt(args['clusters'], max_rows=1, skiprows=1, usecols=(0, 1))
    print(params)

    # get clusters from sklearn
    true_clusters = dbscan(point_cloud, float(params[0]), int(params[1]))

    # load clusters from given file, and convert to monotonically increasing
    pred_clusters = np.loadtxt(args['clusters'], skiprows=2)
    cluster_set = {}
    next_cluster = 0
    for i in range(len(pred_clusters)):
        cluster = pred_clusters[i]
        if cluster in cluster_set:
            pred_clusters[i] = cluster_set[cluster]
        elif cluster != -1:
            cluster_set[cluster] = next_cluster
            next_cluster += 1

    # check for mismatches with clusters from sklearn
    mismatches = np.argwhere(pred_clusters != true_clusters).squeeze()

    for i, mismatch in enumerate(mismatches):
        if not verbose and i >= MISMATCH_REPORT_CNT:
            break
        print("Actual cluster is {}, predicted cluster is {}".format(true_clusters[mismatch], pred_clusters[mismatch]))

    if len(mismatches) > 0:
        return False
    return True

    # TODO: Look at DBSCAN clustering metrics to assess quality


def main(args):
    val = validate(args)
    if val:
        LOG.info('Validate succeeded.')
    else:
        LOG.info('Validation failed.')
        if (not verbose and mismatch_cnt >= MISMATCH_REPORT_CNT):
            LOG.info('Showing truncated report. To see a list of all mismatches, run with \'-v\'.')


if __name__ == '__main__':
    main(parse_args())