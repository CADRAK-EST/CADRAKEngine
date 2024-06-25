import ezdxf
import numpy as np
import re
from collections import defaultdict
from app.parsers.utilities import format_point2, transform_point, apply_transform
from app.parsers.parsing_utilities import (get_entity_color, get_entity_lineweight, get_entity_linetype,
                                           get_entity_layer, get_insert_transform)

# Global cache for storing extracted points
extracted_points_cache = {}


def process_entities(doc_blocks, entities, parent_transform=np.identity(3)):
    points = []
    entity_to_points = defaultdict(list)
    transform_matrices = {}
    border_entities = []
    dimensions = []
    # texts = []

    def process_block(block, transform_matrix):
        block_points = []
        block_entity_to_points = defaultdict(list)
        block_transform_matrices = {}
        block_texts = []
        for entity in block:
            if entity.dxftype() == 'INSERT':
                insert_matrix = get_insert_transform(entity)
                combined_matrix = np.dot(transform_matrix, insert_matrix)
                nested_block = doc_blocks.get(entity.dxf.name)
                if 'border' in entity.dxf.name.lower():
                    border_entities.append((entity, combined_matrix))
                    transform_matrices[entity] = combined_matrix
                    continue
                nested_points, nested_entity_to_points, nested_transform_matrices, _ = process_block(nested_block,
                                                                                                     combined_matrix)
                block_points.extend(nested_points)
                for k, v in nested_entity_to_points.items():
                    block_entity_to_points[k].extend(v)
                block_transform_matrices.update(nested_transform_matrices)
            elif entity.dxftype() == 'TEXT' or entity.dxftype() == 'MTEXT':
                text_data = {
                    "type": entity.dxftype(),
                    "text": entity.dxf.text if entity.dxftype() == 'TEXT' else entity.text,
                    "insert": transform_point(entity.dxf.insert.x, entity.dxf.insert.y, transform_matrix),
                    "height": entity.dxf.height if entity.dxftype() == 'TEXT' else entity.dxf.char_height,
                    "style": entity.dxf.style,
                    "color": "#000000"
                    # "color": get_entity_color(entity, metadata['layer_properties'], metadata['header_defaults'], metadata['background_color'])
                }
                block_texts.append(text_data)
            else:
                entity_points = extract_points_from_entity(entity)
                extracted_points_cache[entity] = entity_points
                if entity_points:
                    transformed_points = apply_transform(transform_matrix, entity_points)
                    block_entity_to_points[entity].extend(transformed_points)
                    block_transform_matrices[entity] = transform_matrix
                    block_points.extend(transformed_points)
        return block_points, block_entity_to_points, block_transform_matrices, block_texts

    for entity in entities:
        if entity.dxftype() == 'INSERT':
            insert_matrix = get_insert_transform(entity)
            block = doc_blocks.get(entity.dxf.name)
            if 'border' in entity.dxf.name.lower():
                border_entities.append((entity, insert_matrix))
                transform_matrices[entity] = insert_matrix
                continue
            block_points, block_entity_to_points, block_transform_matrices, block_texts = process_block(block,
                                                                                                        insert_matrix)
            points.extend(block_points)
            for k, v in block_entity_to_points.items():
                entity_to_points[k].extend(v)
            transform_matrices.update(block_transform_matrices)
        elif entity.dxftype() == 'DIMENSION':
            dimensions.append(entity)
        else:
            entity_points = extract_points_from_entity(entity)
            extracted_points_cache[entity] = entity_points
            if entity_points:
                transformed_points = apply_transform(parent_transform, entity_points)
                entity_to_points[entity].extend(transformed_points)
                transform_matrices[entity] = parent_transform
                points.extend(transformed_points)

    return points, entity_to_points, transform_matrices, border_entities, dimensions


def extract_points_from_entity(entity):
    num_segments = 72
    points = []

    if entity.dxftype() == 'LINE':
        points = [np.array([entity.dxf.start.x, entity.dxf.start.y]), np.array([entity.dxf.end.x, entity.dxf.end.y])]
    elif entity.dxftype() == 'CIRCLE':
        center = np.array([entity.dxf.center.x, entity.dxf.center.y])
        radius = entity.dxf.radius
        angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        points = [center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles]
    elif entity.dxftype() == 'ARC':
        center = np.array([entity.dxf.center.x, entity.dxf.center.y])
        radius = entity.dxf.radius
        start_angle = np.radians(entity.dxf.start_angle)
        end_angle = np.radians(entity.dxf.end_angle)
        angles = np.linspace(start_angle, end_angle, num_segments, endpoint=True)
        points = [center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles]
    elif entity.dxftype() == 'ELLIPSE':
        center = np.array([entity.dxf.center.x, entity.dxf.center.y])
        major_axis = np.array([entity.dxf.major_axis.x, entity.dxf.major_axis.y])
        major_axis_length = np.linalg.norm(major_axis)
        minor_axis_length = major_axis_length * entity.dxf.ratio
        rotation_angle = np.arctan2(major_axis[1], major_axis[0])
        angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        points = [center + np.array([major_axis_length * np.cos(a) * np.cos(rotation_angle) - minor_axis_length * np.sin(a) * np.sin(rotation_angle),
                                     major_axis_length * np.cos(a) * np.sin(rotation_angle) + minor_axis_length * np.sin(a) * np.cos(rotation_angle)]) for a in angles]
    elif entity.dxftype() == 'SPLINE':
        points = [np.array([point.x, point.y]) for point in entity.fit_points]
    elif entity.dxftype() == 'POLYLINE':
        points = [np.array([vertex.dxf.location.x, vertex.dxf.location.y]) for vertex in entity.vertices]
    elif entity.dxftype() == 'HATCH':
        for path in entity.paths:
            if isinstance(path, ezdxf.entities.PolylinePath):
                points.extend([np.array([v.x, v.y]) for v in path.vertices])
            elif isinstance(path, ezdxf.entities.EdgePath):
                for edge in path.edges:
                    if edge.EDGE_TYPE == 'LineEdge':
                        points.extend([np.array([edge.start.x, edge.start.y]), np.array([edge.end.x, edge.end.y])])
                    elif edge.EDGE_TYPE == 'ArcEdge':
                        center = np.array([edge.center.x, edge.center.y])
                        radius = edge.radius
                        start_angle = np.radians(edge.start_angle)
                        end_angle = np.radians(edge.end_angle)
                        angles = np.linspace(start_angle, end_angle, num_segments, endpoint=True)
                        points.extend([center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles])

    return points


def get_entity_points_from_cache(entity):
    return extracted_points_cache[entity]


def classify_entities(cluster, transform_matrices, metadata, layer_properties, header_defaults):
    contours = {"lines": [], "circles": [], "arcs": [], "lwpolylines": [], "polylines": [], "solids": [],
                "ellipses": [], "splines": []}
    for entity in cluster:
        transform_matrix = transform_matrices[entity]
        entity_color = get_entity_color(entity, layer_properties, header_defaults, metadata["background_color"])
        line_weight = get_entity_lineweight(entity, layer_properties, header_defaults)
        line_style = get_entity_linetype(entity, layer_properties, header_defaults)
        layer = get_entity_layer(entity, layer_properties, header_defaults)
        if entity.dxftype() == 'LINE':
            start = format_point2(transform_point(entity.dxf.start.x, entity.dxf.start.y, transform_matrix))
            end = format_point2(transform_point(entity.dxf.end.x, entity.dxf.end.y, transform_matrix))
            contours["lines"].append(
                {"start": start, "end": end, "colour": entity_color, "weight": line_weight,
                 "style": line_style, "layer": layer})
        elif entity.dxftype() == 'CIRCLE':
            center = format_point2(transform_point(entity.dxf.center.x, entity.dxf.center.y, transform_matrix))
            contours["circles"].append(
                {"centre": center, "radius": entity.dxf.radius, "colour": entity_color, "weight": line_weight,
                 "style": line_style, "layer": layer})
        elif entity.dxftype() == 'ARC':
            center = format_point2(transform_point(entity.dxf.center.x, entity.dxf.center.y, transform_matrix))
            contours["arcs"].append(
                {"centre": center, "radius": entity.dxf.radius, "start_angle": entity.dxf.start_angle,
                 "end_angle": entity.dxf.end_angle, "colour": entity_color, "weight": line_weight, "style": line_style,
                 "layer": layer})
        elif entity.dxftype() == 'LWPOLYLINE':
            points = [format_point2(transform_point(p[0], p[1], transform_matrix)) for p in entity.get_points()]
            contours["lwpolylines"].append(
                {"points": points, "colour": entity_color, "weight": line_weight, "style": line_style, "layer": layer})
        elif entity.dxftype() == 'POLYLINE':
            points = [format_point2(transform_point(v.dxf.location.x, v.dxf.location.y, transform_matrix)) for v in
                      entity.vertices]
            contours["polylines"].append(
                {"points": points, "colour": entity_color, "weight": line_weight, "style": line_style, "layer": layer})
        elif entity.dxftype() == 'SOLID':
            points = [format_point2(transform_point(entity.dxf.vtx0.x, entity.dxf.vtx0.y, transform_matrix)),
                      format_point2(transform_point(entity.dxf.vtx1.x, entity.dxf.vtx1.y, transform_matrix)),
                      format_point2(transform_point(entity.dxf.vtx2.x, entity.dxf.vtx2.y, transform_matrix)),
                      format_point2(transform_point(entity.dxf.vtx3.x, entity.dxf.vtx3.y, transform_matrix))]
            contours["solids"].append(
                {"points": points, "colour": entity_color, "weight": line_weight, "style": line_style, "layer": layer})
        elif entity.dxftype() == 'ELLIPSE':
            center = format_point2(transform_point(entity.dxf.center.x, entity.dxf.center.y, transform_matrix))
            major_axis_vector = transform_point(entity.dxf.major_axis[0], entity.dxf.major_axis[1], transform_matrix)
            major_axis_length = np.linalg.norm([major_axis_vector[0], major_axis_vector[1]])
            minor_axis_length = major_axis_length * entity.dxf.ratio
            rotation_angle = np.arctan2(major_axis_vector[1], major_axis_vector[0])
            contours["ellipses"].append({"centre": center, "major_axis_length": major_axis_length * 2,
                                         "minor_axis_length": minor_axis_length * 2,
                                         "rotation_angle": np.degrees(rotation_angle), "colour": entity_color,
                                         "weight": line_weight, "style": line_style, "layer": layer})
        elif entity.dxftype() == 'SPLINE':
            points = [format_point2(transform_point(p[0], p[1], transform_matrix)) for p in entity.fit_points]
            contours["splines"].append(
                {"points": points, "colour": entity_color, "weight": line_weight, "style": line_style, "layer": layer})
    return contours


def classify_text_entities(all_entities, metadata, layer_properties, header_defaults):
    print("All entities: ", all_entities)
    texts = {"texts": [], "mtexts": []}
    for entity in all_entities:
        print("Entities", entity)
        if entity.dxftype() == 'ACIDBLOCKREFERENCE':
            continue
        entity_color = get_entity_color(entity, layer_properties, header_defaults, metadata["background_color"])
        line_weight = get_entity_lineweight(entity, layer_properties, header_defaults)
        line_style = get_entity_linetype(entity, layer_properties, header_defaults)
        layer = get_entity_layer(entity, layer_properties, header_defaults)
        if entity.dxftype() == 'TEXT':
            print("Found a text!")
            text = entity.dxf.text
            height = entity.dxf.height
            style = entity.dxf.style
            texts["texts"].append({"text": text, "height": height, "style": style,
                                   "colour": entity_color, "layer": layer})
        elif entity.dxftype() == 'MTEXT':
            print("Found an mtext!")
            """Strip unnecessary formatting tags from MTEXT content."""
            # Remove font definitions and other formatting tags
            text = re.sub(r'\\f[^;]*;|\\[A-Za-z]+\;|\\H\d+\.\d+;|\\P|{\\H[^}]*;|}', '', entity.text)
            # Remove any remaining braces and other extraneous characters
            text = re.sub(r'{|}', '', text)
            text_center = entity.dxf.insert.x, entity.dxf.insert.y
            height = entity.dxf.char_height
            style = entity.dxf.style
            # color currently manually set to black as the entity_colour of the texts are usually white
            texts["mtexts"].append({"text": text, "center": text_center, "height": height,
                                    "style": style, "colour": "#000000", "layer": layer})
    return texts


def get_doc_data(doc):

    metadata = {
        "filename": "",
        "units": doc.header.get('$INSUNITS', 0),
        "software_version": doc.header.get('$ACADVER', 'Unknown'),
        "background_color": "0xFFFFFF",
        "bounding_box": doc.header.get('$EXTMIN', None)[:2] + doc.header.get('$EXTMAX', None)[:2]
    }

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

    all_entities = list(doc.modelspace())

    return metadata, layer_properties, header_defaults, all_entities


def process_border_block(border_entities, doc_blocks, metadata, layer_properties, header_defaults):
    border_contours = {
        "lines": [], "circles": [], "arcs": [], "lwpolylines": [], "polylines": [], "solids": [], "ellipses": [],
        "splines": []
    }
    for be, matrix in border_entities:
        block = doc_blocks.get(be.dxf.name)
        _, block_entity_to_points, block_transform_matrices, _, _ = process_entities(doc_blocks, list(block), matrix)
        for entity in block_entity_to_points:
            classified = classify_entities([entity], block_transform_matrices, metadata, layer_properties,
                                           header_defaults)
            for key in border_contours.keys():
                border_contours[key].extend(classified.get(key, []))
    border_view = {"contours": border_contours, "block_name": "Border"}

    return border_view
