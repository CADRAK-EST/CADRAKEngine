import ezdxf
import numpy as np
from app.parsers.matplotlib_visualization import plot_entities, indicate_mistakes
import json
import os
from app.parsers.parsing_clustering import process_entities, classify_entities, iterative_merge, \
    assign_entities_to_clusters, classify_text_entities
from sklearn.cluster import DBSCAN
import time
from app.parsers.dimension_analysis import find_lengths
import logging
import cProfile
import pstats

logger = logging.getLogger(__name__)


def read_dxf(file_path):
    doc = ezdxf.readfile(file_path)
    metadata = {
        "filename": os.path.basename(file_path),
        "units": doc.header.get('$INSUNITS', 0),
        "software_version": doc.header.get('$ACADVER', 'Unknown'),
        "background_color": "0xFFFFFF",
        "bounding_box": doc.header.get('$EXTMIN', None)[:2] + doc.header.get('$EXTMAX', None)[:2]
    }
    all_entities = list(doc.modelspace())

    points, entity_to_points, transform_matrices, border_entities, dimensions = process_entities(doc, all_entities,
                                                                                                 metadata)

    for dimension in dimensions:
        #print(dimension)
        corresponding_geometry_block_name = dimension.dxf.get('geometry', None)

    flat_points, labels = form_initial_clusters(entity_to_points)

    #flat_points = np.array([pt for sublist in entity_to_points.values() for pt in sublist])
    #db = DBSCAN(eps=5, min_samples=1).fit(flat_points)
    #labels = db.labels_

    # Exclude border entities from clustering
    clusters = assign_entities_to_clusters(
        {k: v for k, v in entity_to_points.items() if k not in [be[0] for be in border_entities]}, flat_points, labels)

    # Convert clusters dictionary to list of lists
    cluster_list = list(clusters.values())

    # Build layer properties dictionary
    layer_properties = {}
    for layer in doc.layers:
        layer_properties[layer.dxf.name] = {
            "color": layer.dxf.get('color', 256),
            "lineweight": layer.dxf.get('lineweight', -1),
            "linetype": layer.dxf.get('linetype', 'BYLAYER'),
            "name": layer.dxf.get('name', 'noname')
        }

    # Read the default values from the DXF header
    header_defaults = {
        "color": doc.header.get('$CECOLOR', 256),  # Default color
        "lineweight": doc.header.get('$CELWEIGHT', -1),  # Default lineweight
        "linetype": doc.header.get('$CELTYPE', 'BYLAYER')  # Default line type
    }

    # Process border entities
    border_view = None
    if border_entities:
        border_contours = {
            "lines": [], "circles": [], "arcs": [], "lwpolylines": [], "polylines": [], "solids": [], "ellipses": [],
            "splines": []
        }
        for be, matrix in border_entities:
            block = doc.blocks.get(be.dxf.name)
            _, block_entity_to_points, block_transform_matrices, _, _ = process_entities(doc, list(block), metadata,
                                                                                         matrix)
            for entity in block_entity_to_points:
                classified = classify_entities([entity], block_transform_matrices, metadata, layer_properties,
                                               header_defaults)
                for key in border_contours.keys():
                    border_contours[key].extend(classified.get(key, []))
        border_view = {"contours": border_contours, "block_name": "Border"}

    final_clusters = iterative_merge(cluster_list, 5)

    views = [{"contours": classify_entities(cluster, transform_matrices, metadata, layer_properties, header_defaults),
              "block_name": f"View {idx + 1}"} for idx, cluster in enumerate(final_clusters)]
    texts = [classify_text_entities(all_entities, metadata, layer_properties, header_defaults)]

    if border_view:
        views.append(border_view)

    return views, dimensions, metadata, texts


def initialize(file_path, visualize=False, save=False, analyze=True, log_times=True):
    parse_time = time.time()
    views, dimensions, metadata, texts = read_dxf(file_path)
    info_boxes = []
    parse_time = time.time() - parse_time

    if analyze:
        analyze_time = time.time()
        views = mistake_analysis(views, dimensions)
        indicate_mistakes(views)
        analyze_time = time.time() - analyze_time

    page = {"metadata": metadata, "bounding_box": {}, "views": views, "info_boxes": [], "texts": texts}

    if visualize:
        visualize_time = time.time()
        plot_entities(views, info_boxes)
        visualize_time = time.time() - visualize_time

    if save:
        save_time = time.time()
        save_json(page)
        save_time = time.time() - save_time

    if log_times:
        logger.info(f"Time taken for parsing: {parse_time:.2f}s")
        if visualize:
            logger.info(f"Time taken for visualization: {visualize_time:.2f}s")
        if analyze:
            logger.info(f"Time taken for analysis: {analyze_time:.2f}s")
        if save:
            logger.info(f"Time taken for saving: {save_time:.2f}s")
    return page


def mistake_analysis(views, dimensions):
    for view in views:
        ids_of_mistaken_lines, ids_of_potential_mistaken_lines = find_lengths(dimensions, view["contours"]["lines"], view["contours"]["circles"])
        view["mistakes"] = {"potential": {}, "certain": {}}
        view["mistakes"]["certain"]["lines"] = ids_of_mistaken_lines
        view["mistakes"]["potential"]["lines"] = ids_of_potential_mistaken_lines
        logger.info(f"{view['block_name']} has {len(ids_of_mistaken_lines)} mistakes and {len(ids_of_potential_mistaken_lines)} potential mistakes.")
    return views


def save_json(page):
    with open("testing_data.json", "w") as f:
        json.dump(page, f, indent=4)


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
    return flat_points, db.fit_predict(flat_points)


if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), "../../test_data", "12-04-0 Kiik SynDat 3/12-04-0 Kiik SynDat 3_Sheet_1.dxf")

    profile = False

    if profile:
        # Create a profile object
        pr = cProfile.Profile()
        pr.enable()

        initialize(file_path, visualize=True, save=False, analyze=True, log_times=True)

        pr.disable()

        # Save profiling results to a file
        with open("profiling_results.prof", "w") as f:
            ps = pstats.Stats(pr, stream=f)
            ps.strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats()

    # Optionally, print profiling results to the console
    #ps = pstats.Stats(pr)
    #ps.strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats()
