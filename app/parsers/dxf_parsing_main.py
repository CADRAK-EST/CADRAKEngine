import ezdxf
import json
import os
import logging
import cProfile
import pstats
import time
from app.parsers.parsing import (process_entities, classify_entities, classify_text_entities, get_doc_data,
                                 process_border_block)
from app.parsers.clustering import iterative_merge, assign_entities_to_clusters, form_initial_clusters
from app.parsers.visualization_utilities import plot_entities, indicate_mistakes
from app.parsers.dimension_analysis import find_lengths

logger = logging.getLogger(__name__)


def read_dxf(file_path):
    doc = ezdxf.readfile(file_path)
    doc_blocks = doc.blocks
    metadata, layer_properties, header_defaults, all_entities = get_doc_data(doc)

    points, entity_to_points, transform_matrices, border_entities, dimensions, texts = process_entities(doc.blocks, all_entities)

    print("\n1")
    print(texts)

    for dimension in dimensions:
        #print(dimension)
        corresponding_geometry_block_name = dimension.dxf.get('geometry', None)

    flat_points, labels = form_initial_clusters(entity_to_points)

    # Exclude border entities from clustering
    clusters = assign_entities_to_clusters(
        {k: v for k, v in entity_to_points.items() if k not in [be[0] for be in border_entities]}, flat_points, labels)

    # Convert clusters dictionary to list of lists
    cluster_list = list(clusters.values())

    # Process border entities
    border_view = None
    if border_entities:
        border_view = process_border_block(border_entities, doc_blocks, metadata, layer_properties, header_defaults)

    final_clusters = iterative_merge(cluster_list, 5)
    # final_clusters = cluster_list

    views = [{"contours": classify_entities(cluster, transform_matrices, entity_to_points,metadata, layer_properties, header_defaults),
              "block_name": f"View {idx + 1}"} for idx, cluster in enumerate(final_clusters)]

    text_entities = classify_text_entities(all_entities, metadata, layer_properties, header_defaults)

    # Merge texts from both process_entities and classify_text_entities
    texts['texts'].extend(text_entities['texts'])
    texts['mtexts'].extend(text_entities['mtexts'])

    if border_view:
       views.append(border_view)

    return views, dimensions, metadata, texts


def initialize(file_path, visualize=False, save=False, analyze=True, log_times=True):
    parse_time = time.time()
    views, dimensions, metadata, all_texts = read_dxf(file_path)
    info_boxes = []
    parse_time = time.time() - parse_time

    if analyze:
        analyze_time = time.time()
        views = mistake_analysis(views, dimensions)
        indicate_mistakes(views)
        analyze_time = time.time() - analyze_time

    page = {"metadata": metadata, "bounding_box": {}, "views": views, "info_boxes": [], "texts": all_texts}

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
        ids_of_mistaken_lines, ids_of_potential_mistaken_lines = find_lengths(dimensions, view["contours"]["lines"],
                                                                              view["contours"]["circles"])
        view["mistakes"] = {"potential": {}, "certain": {}}
        view["mistakes"]["certain"]["lines"] = ids_of_mistaken_lines
        view["mistakes"]["potential"]["lines"] = ids_of_potential_mistaken_lines
        logger.info(f"{view['block_name']} has {len(ids_of_mistaken_lines)} mistakes and "
                    f"{len(ids_of_potential_mistaken_lines)} potential mistakes.")
    return views


def save_json(page):
    with open("testing_data.json", "w") as json_file:
        json.dump(page, json_file, indent=4)


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
        # ps = pstats.Stats(pr)
        # ps.strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats()
    else:
        initialize(file_path, visualize=True, save=False, analyze=True, log_times=True)
