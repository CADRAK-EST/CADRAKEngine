from sklearn.cluster import DBSCAN
from shapely.strtree import STRtree
import alphashape
from shapely.geometry import MultiPoint
from collections import defaultdict
from scipy.spatial import KDTree
import numpy as np
from app.parsers.parsing import get_entity_points_from_cache


def assign_entities_to_clusters(entity_to_points, points, labels):
    clusters = defaultdict(list)
    point_tree = KDTree(points)

    for entity, entity_points in entity_to_points.items():
        for ep in entity_points:
            distance, index = point_tree.query(ep[:2], k=1)
            cluster_id = labels[index]
            clusters[cluster_id].append(entity)
            break  # Each entity is assigned to the cluster of its first point

    return clusters


def form_initial_clusters(entity_to_points):
    # Step 1: Flatten the dictionary to a list of points and a list of initial labels
    flat_points = []
    initial_labels = []
    for label, sublist in enumerate(entity_to_points.values()):
        flat_points.extend(sublist)
        initial_labels.extend([label] * len(sublist))

    flat_points = np.array(flat_points)
    initial_labels = np.array(initial_labels)

    # Step 2: Custom DBSCAN class with optimized distance checking
    class CustomDBSCAN(DBSCAN):
        def fit(self, X):
            self.X = X
            self.initial_labels = initial_labels
            return super().fit(X)

        def _region_query(self, point_idx):
            point = self.X[point_idx]
            neighbors = []
            for idx in range(self.X.shape[0]):
                if self.initial_labels[point_idx] == self.initial_labels[idx] or np.linalg.norm(
                        point - self.X[idx]) <= self.eps:
                    neighbors.append(idx)
            return neighbors

        def fit_predict(self, X, y=None, sample_weight=None):
            self.fit(X)
            labels = self.labels_
            return labels

    # Step 3: Run the optimized CustomDBSCAN
    db = CustomDBSCAN(eps=5, min_samples=1)
    return flat_points, db.fit_predict(flat_points)  # initial_labels, db.fit_predict(flat_points)


def merge_clusters_with_alpha_shape(clusters, alpha, alpha_shapes):
    merged = set()
    new_alpha_shapes = {}
    cluster_mapping = {}

    # Rebuild spatial index
    shapes = [alpha_shapes[idx] for idx in alpha_shapes]
    tree = STRtree(shapes)

    for idx1, alpha_shape1 in list(alpha_shapes.items()):
        if idx1 in merged:
            continue

        merged_current = False
        for idx2 in tree.query(alpha_shape1):
            if idx1 == idx2 or idx2 in merged:
                continue

            alpha_shape2 = alpha_shapes[idx2]

            if alpha_shape1.intersects(alpha_shape2):
                # Merge clusters
                #print(clusters[idx1])
                #print(clusters[idx2])
                new_cluster = clusters[idx1] + clusters[idx2]
                #print(new_cluster)
                merged.add(idx1)
                merged.add(idx2)

                # Assign a new index for the new cluster
                new_idx = len(new_alpha_shapes)
                new_alpha_shapes[new_idx] = get_alpha_shape(new_cluster, alpha)
                cluster_mapping[new_idx] = new_cluster
                merged_current = True
                # print(f"Merged clusters {idx1} and {idx2} into new cluster {new_idx}.")
                break

        if not merged_current:
            new_idx = len(new_alpha_shapes)
            new_alpha_shapes[new_idx] = alpha_shape1
            cluster_mapping[new_idx] = clusters[idx1]

    # Convert mapping to a list
    new_clusters = [cluster_mapping[idx] for idx in sorted(cluster_mapping.keys())]
    alpha_shapes = new_alpha_shapes

    return new_clusters, alpha_shapes


def iterative_merge(clusters, alpha):
    iterations = 0
    alpha_shapes = {idx: get_alpha_shape(cluster, alpha) for idx, cluster in enumerate(clusters)}

    while True:
        num_clusters_before = len(clusters)
        print(f"Iteration {iterations}: {num_clusters_before} clusters before merge.")
        clusters, alpha_shapes = merge_clusters_with_alpha_shape(clusters, alpha, alpha_shapes)

        if num_clusters_before == len(clusters):
            break
        iterations += 1

    return clusters


def entities_to_points(cluster):
    points = []
    for entity in cluster:
        points.extend(get_entity_points_from_cache(entity))
    return points


def get_alpha_shape(cluster, alpha=0.1):
    points = entities_to_points(cluster)
    """
    try:
        return alphashape.alphashape(points, alpha)
    except Exception:
        return MultiPoint(points).convex_hull
    """
    return MultiPoint(points).convex_hull
    points_array = np.array(points)
    if np.all(points_array[:, 0] == points_array[0, 0]) or np.all(points_array[:, 1] == points_array[0, 1]):
        print("Degenerate point set (all x or all y coordinates are the same), using convex hull.")
        return MultiPoint(points).convex_hull

    return alphashape.alphashape(points, alpha)
