import ezdxf
import numpy as np
from app.parsers.utilities import normalize_point2
from app.parsers.visualization import plot_entities
import json
import os
from collections import defaultdict
from app.parsers.clustering import process_entities, classify_entities, iterative_merge
from sklearn.cluster import DBSCAN
import time
import logging

logger = logging.getLogger(__name__)

def read_dxf(file_path):
    doc = ezdxf.readfile(file_path)
    metadata = {
        "filename": os.path.basename(file_path),
        "units": doc.header.get('$INSUNITS', 0),
        "software_version": doc.header.get('$ACADVER', 'Unknown'),
        "background_color": "0xFFFFFF",
        "bounding_box": doc.header.get('$EXTMIN', (0, 0, 0)) + doc.header.get('$EXTMAX', (0, 0, 0))
    }
    all_entities = list(doc.modelspace())

    points, entity_to_points, transform_matrices = process_entities(doc, all_entities, metadata)

    flat_points = np.array([pt for sublist in entity_to_points.values() for pt in sublist])
    db = DBSCAN(eps=5, min_samples=1).fit(flat_points)
    labels = db.labels_

    point_to_cluster = {tuple(pt): labels[idx] for idx, pt in enumerate(flat_points)}

    clusters = defaultdict(list)
    for entity, pts in entity_to_points.items():
        cluster_ids = set(point_to_cluster[tuple(pt)] for pt in pts if tuple(pt) in point_to_cluster)
        for cluster_id in cluster_ids:
            clusters[cluster_id].append(entity)

    final_clusters = iterative_merge(list(clusters.values()), 5)

    views = [{"contours": classify_entities(cluster, transform_matrices, metadata), "block_name": f"View {idx + 1}"} for idx, cluster in enumerate(final_clusters)]

    return views, [], metadata

def initialize(file_path, visualize=False, save=True, analyze=True):
    start_time = time.time()
    views, info_boxes, metadata = read_dxf(file_path)
    page = {"metadata": metadata, "bounding_box": {}, "views": views, "info_boxes": info_boxes}
    logger.info(f"Time taken for parsing: {time.time() - start_time:.2f}s")
    if visualize:
        start_time = time.time()
        plot_entities(views, info_boxes)
        logger.info(f"Time taken for visualization: {time.time() - start_time:.2f}s")
    if analyze:
        start_time = time.time()
        mistake_analysis(views, info_boxes)
        logger.info(f"Time taken for analysis: {time.time() - start_time:.2f}s")
    if save:
        start_time = time.time()
        save_json(page)
        logger.info(f"Time taken for saving: {time.time() - start_time:.2f}s")
    
    return page

def mistake_analysis(views, info_boxes):
    pass

def save_json(page):
    with open("data.json", "w") as f:
        json.dump(page, f, indent=4)

if __name__ == "__main__":
    file_path = "data/LauriToru.dxf"
    initialize(file_path, True, True, False)
