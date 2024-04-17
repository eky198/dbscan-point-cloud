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
  -e EPSILON            Maximum Euclidean distance for point to be considered a neighbor
  -p MIN_PTS            Minimum number of neighboring points to form a cluster or core point
  -d DATA               Raw point cloud data file
  -c CLUSTERS           Cluster index for each point (ordered w.r.t. data file)
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
    if '-e' not in args or 'p' not in args or '-d' not in args or '-c' not in args:
        print (help_message)
        sys.exit(1)
    if '-v' in args:
        verbose = True
    parsed = {}
    parsed['epsilon'] = args[args.index('-e') + 1]
    parsed['min_pts'] = args[args.index('-p') + 1]
    parsed['data'] = args[args.index('-r') + 1]
    parsed['clusters'] = args[args.index('-c') + 1]
    return parsed


def validate(args):
    global mismatch_cnt

    point_cloud = np.fromfile(args['data'], '<f4')
    point_cloud = np.reshape(point_cloud, (-1, 4))  
    true_clusters = dbscan(point_cloud, args['epsilon'], args['min_pts'])

    # TODO: Read predicted clusters from args['clusters'] txt file
    # TODO: Compare true clusters to predicted clusters, keeping track of any mismatches
    # TODO: If verbose, print out all the mismatched points
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