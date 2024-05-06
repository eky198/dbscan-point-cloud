# import pptk
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt

FORMAT = '%(asctime)-15s - %(levelname)s - %(module)10s:%(lineno)-5d - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)
LOG = logging.getLogger(__name__)

help_message = '''
usage: validate.py [-h] [-d point cloud raw data file] [-c clusters file]

Visualize the DBSCAN point cloud clusters

optional arguments:
  -h, --help            Show this help message and exit
  -v, --verbose         Show all mismatched labels
'''


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


# def visualize_pptk(point_cloud, pred_clusters):
#     viewer = pptk.viewer(point_cloud[:, :3])
#     viewer.attributes(pred_clusters)
#     viewer.set(point_size=1)


def visualize(point_cloud, pred_clusters):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=pred_clusters)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()  


def main(args):
    # load point cloud from text file
    point_cloud = np.loadtxt(args['data'])
    point_cloud = np.reshape(point_cloud, (-1, 4))
    print(point_cloud)
    params = np.loadtxt(args['clusters'], max_rows=1, skiprows=1, usecols=(0, 1))
    print(params)

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

    visualize(point_cloud, pred_clusters)


if __name__ == '__main__':
    main(parse_args())