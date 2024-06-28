import ezdxf
import json
import os
import logging
import cProfile
import pstats
import time
from app.parsers.parsing import process_entities, classify_entities, classify_text_entities, process_border_block
from app.parsers.parsing_utilities import get_doc_data
from app.parsers.clustering import iterative_merge, assign_entities_to_clusters, form_initial_clusters, find_closest_view
from app.parsers.visualization_utilities import plot_entities, indicate_mistakes
from app.parsers.dimension_analysis import find_lengths
from app.parsers.text_analysis.text_analysis import analyze_texts

logger = logging.getLogger(__name__)


def get_text_styles(doc):
    text_styles = {}
    for style in doc.styles:
        text_styles[style.dxf.name] = style.dxf.font
    return text_styles

def assign_views_to_texts(texts, alpha_shapes):
    for text_type in ['texts', 'mtexts', 'attdefs']:
        for text in texts[text_type]:
            text_center = text['center']
            closest_view = find_closest_view(text_center, alpha_shapes)
            print("Closest view", closest_view)
            text['view'] = closest_view
    return texts


def read_dxf(file_path):
    doc = ezdxf.readfile(file_path)
    doc_blocks = doc.blocks
    metadata, layer_properties, header_defaults, all_entities = get_doc_data(doc)

    text_styles = get_text_styles(doc)

    points, entity_to_points, transform_matrices, border_inserts, dimensions, texts = process_entities(doc.blocks, all_entities, text_styles)

    for dimension in dimensions:
        pass

    flat_points, labels = form_initial_clusters(entity_to_points)

    # Exclude border entities from clustering
    clusters = assign_entities_to_clusters(
        {k: v for k, v in entity_to_points.items() if k not in [be[0] for be in border_inserts]}, flat_points, labels)

    # Convert clusters dictionary to list of lists
    cluster_list = list(clusters.values())
    print("cluster_list: ")
    print(cluster_list)

    # Process border entities
    border_view = None
    if len(border_inserts) > 0:
        border_view = process_border_block(border_inserts, doc_blocks, metadata, layer_properties, header_defaults)
        pass

    final_clusters, alpha_shapes = iterative_merge(cluster_list, 0)


    text_entities = classify_text_entities(all_entities, text_styles, metadata, layer_properties, header_defaults)

    # Merge texts from both process_entities and classify_text_entities
    texts['texts'].extend(text_entities['texts'])
    texts['mtexts'].extend(text_entities['mtexts'])
    texts['attdefs'].extend(text_entities['attdefs'])

    # Assign views to texts
    texts_with_views = assign_views_to_texts(texts, alpha_shapes)

    # Debug: Print texts_with_views to check its structure
    print("texts_with_views:", texts_with_views)

    views = [
        {
            "contours": classify_entities(cluster, transform_matrices, entity_to_points,metadata, layer_properties, header_defaults),
            "block_name": f"View {idx + 1}",
            "texts": [
                text for text_type in texts_with_views for text in texts_with_views[text_type] if text.get("view") == idx
            ]
        } 
        for idx, cluster in enumerate(final_clusters)]

    if border_view:
       views.append(border_view)
    return views, dimensions, metadata, texts


def initialize(file_path, matplotlib=False, save=False, analyze_dimensions=True, log_times=True, do_analyze_texts=True):
    parse_time = time.time()
    views, dimensions, metadata, all_texts = read_dxf(file_path)
    info_boxes = []
    parse_time = time.time() - parse_time

    if analyze_dimensions:
        analyze_dimensions_time = time.time()
        views = mistake_analysis(views, dimensions)
        indicate_mistakes(views)
        analyze_dimensions_time = time.time() - analyze_dimensions_time

    if do_analyze_texts:
        analyze_texts_time = time.time()
        all_texts = analyze_texts(all_texts)
        analyze_texts_time = time.time() - analyze_texts_time


    page = {"metadata": metadata, "bounding_box": {}, "views": views, "info_boxes": []}

    if matplotlib:
        visualize_time = time.time()
        plot_entities(views, info_boxes)
        visualize_time = time.time() - visualize_time

    if save:
        save_time = time.time()
        save_json(page)
        save_time = time.time() - save_time

    if log_times:
        logger.info(f"Time taken for parsing and clustering: {parse_time:.2f}s")
        if matplotlib:
            logger.info(f"Time taken for visualization: {visualize_time:.2f}s")
        if analyze_dimensions:
            logger.info(f"Time taken for analysis: {analyze_dimensions_time:.4f}s")
        if do_analyze_texts:
            logger.info(f"Time taken for text analysis: {analyze_texts_time:.4f}s")
        if save:
            logger.info(f"Time taken for saving: {save_time:.2f}s")
    return page

def mistake_analysis(views, dimensions):
    for view in views:
        ids_of_mistaken_lines, ids_of_potential_mistaken_lines, ids_of_mistaken_circles =\
            find_lengths(dimensions, view["contours"]["lines"], view["contours"]["circles"])
        view["mistakes"] = {"potential": {}, "certain": {}}
        view["mistakes"]["certain"]["lines"] = ids_of_mistaken_lines
        view["mistakes"]["potential"]["lines"] = ids_of_potential_mistaken_lines
        view["mistakes"]["certain"]["circles"] = ids_of_mistaken_circles
        logger.info(f"{view['block_name']} has {len(ids_of_mistaken_lines)} mistakes and "
                    f"{len(ids_of_potential_mistaken_lines)} potential mistakes.")
    return views


def save_json(page):
    with open("testing_data.json", "w") as json_file:
        json.dump(page, json_file, indent=4)


if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), "../../test_data",
                             "12-02-0 Tiisel CNC SynDat 3/12-02-0 Tiisel CNC SynDat 3_Sheet_4.dxf")

    profile = False

    if profile:
        # Create a profile object
        pr = cProfile.Profile()
        pr.enable()

        initialize(file_path, matplotlib=True, save=False, analyze_dimensions=True, log_times=True, do_analyze_texts=True)

        pr.disable()

        # Save profiling results to a file
        with open("profiling_results.prof", "w") as f:
            ps = pstats.Stats(pr, stream=f)
            ps.strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats()

        # Optionally, print profiling results to the console
        # ps = pstats.Stats(pr)
        # ps.strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats()
    else:
        initialize(file_path, matplotlib=True, save=False, analyze_dimensions=True, log_times=True, do_analyze_texts=False)
