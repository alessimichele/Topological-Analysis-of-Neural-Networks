# mapper algorithm
import kmapper as km
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime
import numpy as np


def compute_mapper(data, title="", n_cubes=1, clusterer=KMeans(n_clusters=10), cube=0):
    current_time = datetime.datetime.now()
    # formatted_time = current_time.strftime("%Y_%d_%m_%H_%M")

    mapper = km.KeplerMapper(verbose=1)

    # Fit to and transform the data
    projected_data = mapper.fit_transform(
        data,
        projection="max",  # to use the max as filtration function
        distance_matrix=None,
        scaler=StandardScaler(),
    )

    # Create dictionary called 'graph' with nodes, edges and meta-information
    graph = mapper.map(
        projected_data,
        data,
        cover=km.Cover(n_cubes=n_cubes),
        clusterer=clusterer,
    )

    # Visualize it
    mapper.visualize(
        graph,
        path_html=f"mapper_outputs/{current_time}.html",
        title=title,
    )

    data_from_cluster = []
    for i in range(10):
        data_from_cluster.append(
            mapper.data_from_cluster_id(
                cluster_id=f"cube{cube}_cluster{i}", graph=graph, data=data
            )
        )

    indexes_from_cluster = []
    for i in range(10):
        cluster = data_from_cluster[i]
        indexes = []
        for j in range(cluster.shape[0]):
            row = cluster[j]
            idx = np.where((row == data))[0][0]
            indexes.append(idx)
        indexes_from_cluster.append(indexes)

    return indexes_from_cluster, data_from_cluster


"""# mapper algorithm
import kmapper as km
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime
import numpy as np


def compute_mapper(data, title="", n_clusters=10):
    current_time = datetime.datetime.now()
    # formatted_time = current_time.strftime("%Y_%d_%m_%H_%M")

    mapper = km.KeplerMapper(verbose=1)

    # Fit to and transform the data
    projected_data = mapper.fit_transform(
        data,
        projection="max",  # to use the max as filtration function
        distance_matrix=None,
        scaler=StandardScaler(),
    )

    # Create dictionary called 'graph' with nodes, edges and meta-information
    graph = mapper.map(
        projected_data,
        data,
        cover=km.Cover(n_cubes=1),
        clusterer=KMeans(n_clusters=n_clusters),
    )

    # Visualize it
    mapper.visualize(
        graph,
        path_html=f"mapper_outputs/{current_time}.html",
        title=title,
    )

    data_from_cluster = []
    for i in range(1, 11):
        data_from_cluster.append(
            mapper.data_from_cluster_id(
                cluster_id=f"cube0_cluster{i}", graph=graph, data=data
            )
        )

    return data_from_cluster
"""
