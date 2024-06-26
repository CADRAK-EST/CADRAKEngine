﻿import ezdxf
import numpy as np
import re
from collections import defaultdict

from shapely import Point

from app.parsers.utilities import (format_point2, transform_point_to_tuple, transform_point_to_list, normalize_vector,
                                   compute_rotation_matrix, transform_height)
from app.parsers.parsing_utilities import (get_entity_color, get_entity_lineweight, get_entity_linetype,
                                           get_entity_layer, get_insert_transform)

# Global cache for storing extracted points
extracted_points_cache = {}


def get_font_for_style(style_name, text_styles):
    if style_name in text_styles:
        return text_styles[style_name]
    return None


def process_entities(doc_blocks, entities, text_styles, parent_transform=np.identity(3)):
    points = []
    entity_to_points = defaultdict(list)
    transform_matrices = {}
    border_inserts = []
    dimensions = []

    texts = {'texts': [], 'mtexts': [], 'attdefs': []}

    def process_block(block, transform_matrix):
        block_points = []
        block_entity_to_points = defaultdict(list)
        block_transform_matrices = {}
        block_texts = {'texts': [], 'mtexts': [], 'attdefs': []}
        for entity in block:
            if entity.dxftype() == 'INSERT':
                insert_matrix = get_insert_transform(entity)
                combined_matrix = np.dot(transform_matrix, insert_matrix)
                nested_block = doc_blocks.get(entity.dxf.name)
                if 'border' in entity.dxf.name.lower():
                    border_inserts.append((entity, combined_matrix))
                    transform_matrices[entity] = combined_matrix
                    continue

                nested_points, nested_entity_to_points, nested_transform_matrices, nested_texts = process_block(nested_block, combined_matrix)

                block_points.extend(nested_points)
                for k, v in nested_entity_to_points.items():
                    block_entity_to_points[k].extend(v)
                block_transform_matrices.update(nested_transform_matrices)
                for text_type, text_list in nested_texts.items():
                    block_texts[text_type].extend(text_list)
            elif entity.dxftype() == 'TEXT':
                text_center = transform_point_to_tuple(entity.dxf.insert.x, entity.dxf.insert.y, transform_matrix)
                text_height = transform_height(entity.dxf.height, transform_matrix)
                font = get_font_for_style(entity.dxf.style, text_styles)
                text_data = {
                    "text": entity.dxf.text,
                    "center": text_center,
                    "height": text_height,
                    "style": entity.dxf.style,
                    "font": font,
                    "color": "#000000"
                }
                block_texts['texts'].append(text_data)
            elif entity.dxftype() == 'MTEXT':
                text_center = transform_point_to_tuple(entity.dxf.insert.x, entity.dxf.insert.y, transform_matrix)
                text_height = transform_height(entity.dxf.char_height, transform_matrix)
                text = re.sub(r'\\f[^;]*;|\\[A-Za-z]+\;|\\H\d+\.\d+;|\\P|{\\H[^}]*;|}|{|}|\\W\d+\.\d+;|\\pxa\d+\.\d+,t\d+;', '', entity.text)
                text_direction = list(entity.dxf.text_direction)
                font = get_font_for_style(entity.dxf.style, text_styles)
                text_data = {
                    "text": text,
                    "center": text_center,
                    "text_direction": text_direction,
                    "attachment_point": entity.dxf.attachment_point,
                    "height": text_height,
                    "style": entity.dxf.style,
                    "font": font,
                    "color": "#000000"
                }
                block_texts['mtexts'].append(text_data)
            elif entity.dxftype() == "ATTDEF":
                text_center = transform_point_to_tuple(entity.dxf.insert.x, entity.dxf.insert.y, transform_matrix)
                text_height = transform_height(entity.dxf.height, transform_matrix)
                font = get_font_for_style(entity.dxf.style, text_styles)
                text_data = {
                    "text": entity.dxf.text,
                    "center": text_center,
                    "height": text_height,
                    "style": entity.dxf.style,
                    "font": font,
                    "color": "#000000",
                }
                block_texts['attdefs'].append(text_data)
            else:
                entity_points = extract_and_transform_points_from_entity(entity, transform_matrix)
                extracted_points_cache[entity] = entity_points
                if entity_points:
                    block_entity_to_points[entity].extend(entity_points)
                    block_transform_matrices[entity] = transform_matrix
                    block_points.extend(entity_points)
        return block_points, block_entity_to_points, block_transform_matrices, block_texts

    for entity in entities:
        if entity.dxftype() == 'INSERT':
            insert_matrix = get_insert_transform(entity)
            block = doc_blocks.get(entity.dxf.name)
            if 'border' in entity.dxf.name.lower():
                border_inserts.append((entity, insert_matrix))
                transform_matrices[entity] = insert_matrix
                continue
            block_points, block_entity_to_points, block_transform_matrices, block_texts = process_block(block,
                                                                                                        insert_matrix)
            points.extend(block_points)
            for k, v in block_entity_to_points.items():
                entity_to_points[k].extend(v)
            transform_matrices.update(block_transform_matrices)
            for text_type, text_list in block_texts.items():
                texts[text_type].extend(text_list)
        elif entity.dxftype() == 'DIMENSION':
            dimensions.append(entity)
        else:
            entity_points = extract_and_transform_points_from_entity(entity, parent_transform)
            extracted_points_cache[entity] = entity_points
            if entity_points:
                entity_to_points[entity].extend(entity_points)
                transform_matrices[entity] = parent_transform
                points.extend(entity_points)

    return points, entity_to_points, transform_matrices, border_inserts, dimensions, texts


def extract_and_transform_points_from_entity(entity, matrix=np.identity(3)):
    num_segments = 72
    points = []
    if entity.dxf.hasattr("extrusion") and entity.dxf.extrusion != (0, 0, 1):
        # normalized_extrusion = normalize_vector(entity.dxf.extrusion)
        # rotational_matrix = compute_rotation_matrix((0, 0, 1), normalized_extrusion)
        # Combine the rotational matrix with the provided transformation matrix
        # matrix = np.dot(matrix, rotational_matrix)
        print("Extrusion not supported")
    if entity.dxftype() == 'LINE':
        points = [np.array(transform_point_to_list(entity.dxf.start.x, entity.dxf.start.y, matrix)),
                  np.array(transform_point_to_list(entity.dxf.end.x, entity.dxf.end.y, matrix))]
    elif entity.dxftype() == 'CIRCLE':
        center = np.array(transform_point_to_list(entity.dxf.center.x if entity.dxf.center.x > 0 else -entity.dxf.center.x, entity.dxf.center.y, matrix))
        radius = transform_height(entity.dxf.radius, matrix)
        angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        points = [center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles]
    elif entity.dxftype() == 'ARC':
        center = np.array(transform_point_to_list(entity.dxf.center.x, entity.dxf.center.y, matrix))
        radius = transform_height(entity.dxf.radius, matrix)
        start_angle = np.radians(entity.dxf.start_angle)
        end_angle = np.radians(entity.dxf.end_angle)
        angles = np.linspace(start_angle, end_angle, num_segments, endpoint=True)
        points = [center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles]
    elif entity.dxftype() == 'ELLIPSE':
        center = np.array(transform_point_to_list(entity.dxf.center.x, entity.dxf.center.y, matrix))
        major_axis = np.array(transform_point_to_list(entity.dxf.major_axis.x, entity.dxf.major_axis.y, matrix))
        major_axis_length = np.linalg.norm(major_axis)
        minor_axis_length = major_axis_length * entity.dxf.ratio
        rotation_angle = np.arctan2(major_axis[1], major_axis[0])
        angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        points = [center + np.array([major_axis_length * np.cos(a) * np.cos(rotation_angle) - minor_axis_length * np.sin(a) * np.sin(rotation_angle),
                                     major_axis_length * np.cos(a) * np.sin(rotation_angle) + minor_axis_length * np.sin(a) * np.cos(rotation_angle)]) for a in angles]
    elif entity.dxftype() == 'SPLINE':
        points = [np.array(transform_point_to_list(point.x, point.y, matrix)) for point in entity.fit_points]
    elif entity.dxftype() == 'LWPOLYLINE':
        points = [np.array(transform_point_to_list(vertex.x, vertex.y, matrix)) for vertex in entity]
    elif entity.dxftype() == 'POLYLINE':
        points = [np.array(transform_point_to_list(vertex.dxf.location.x, vertex.dxf.location.y, matrix)) for vertex in entity.vertices]
    elif entity.dxftype() == 'SOLID':
        points = [np.array(transform_point_to_list(entity.dxf.vtx0.x, entity.dxf.vtx0.y, matrix)),
                  np.array(transform_point_to_list(entity.dxf.vtx1.x, entity.dxf.vtx1.y, matrix)),
                  np.array(transform_point_to_list(entity.dxf.vtx2.x, entity.dxf.vtx2.y, matrix)),
                  np.array(transform_point_to_list(entity.dxf.vtx3.x, entity.dxf.vtx3.y, matrix))]
    elif entity.dxftype() == 'HATCH':
        for path in entity.paths:
            if isinstance(path, ezdxf.entities.PolylinePath):
                points.extend([np.array(transform_point_to_list(v.x, v.y, matrix)) for v in path.vertices])
            elif isinstance(path, ezdxf.entities.EdgePath):
                for edge in path.edges:
                    if edge.EDGE_TYPE == 'LineEdge':
                        points.extend([np.array(transform_point_to_list(edge.start.x, edge.start.y, matrix)),
                                       np.array(transform_point_to_list(edge.end.x, edge.end.y, matrix))])
                    elif edge.EDGE_TYPE == 'ArcEdge':
                        center = np.array(transform_point_to_list(edge.center.x, edge.center.y, matrix))
                        radius = transform_height(edge.radius, matrix) #edge.radius
                        start_angle = np.radians(edge.start_angle)
                        end_angle = np.radians(edge.end_angle)
                        angles = np.linspace(start_angle, end_angle, num_segments, endpoint=True)
                        points.extend([center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles])

    return points


def extract_points_from_entity(entity):
    num_segments = 72
    points = []
    if entity.dxftype() == 'LINE':
        points = [np.array([entity.dxf.start.x, entity.dxf.start.y]),
                  np.array([entity.dxf.end.x, entity.dxf.end.y])]
    elif entity.dxftype() == 'CIRCLE':
        center = np.array([entity.dxf.center.x if entity.dxf.center.x > 0 else -entity.dxf.center.x, entity.dxf.center.y])
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
    elif entity.dxftype() == 'LWPOLYLINE':
        points = [np.array([vertex.x, vertex.y]) for vertex in entity]
    elif entity.dxftype() == 'POLYLINE':
        points = [np.array([vertex.dxf.location.x, vertex.dxf.location.y]) for vertex in entity.vertices]
    elif entity.dxftype() == 'SOLID':
        points = [np.array([entity.dxf.vtx0.x, entity.dxf.vtx0.y]),
                  np.array([entity.dxf.vtx1.x, entity.dxf.vtx1.y]),
                  np.array([entity.dxf.vtx2.x, entity.dxf.vtx2.y]),
                  np.array([entity.dxf.vtx3.x, entity.dxf.vtx3.y])]
    elif entity.dxftype() == 'HATCH':
        for path in entity.paths:
            if isinstance(path, ezdxf.entities.PolylinePath):
                points.extend([np.array([v.x, v.y]) for v in path.vertices])
            elif isinstance(path, ezdxf.entities.EdgePath):
                for edge in path.edges:
                    if edge.EDGE_TYPE == 'LineEdge':
                        points.extend([np.array([edge.start.x, edge.start.y]),
                                       np.array([edge.end.x, edge.end.y])])
                    elif edge.EDGE_TYPE == 'ArcEdge':
                        center = np.array([edge.center.x, edge.center.y])
                        radius = edge.radius #edge.radius
                        start_angle = np.radians(edge.start_angle)
                        end_angle = np.radians(edge.end_angle)
                        angles = np.linspace(start_angle, end_angle, num_segments, endpoint=True)
                        points.extend([center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles])

    return points


def get_entity_points_from_cache(entity):
    return extracted_points_cache[entity]


def classify_contour_entities(cluster, transform_matrices, entity_to_points, metadata, layer_properties, header_defaults):
    contours = {"lines": [], "circles": [], "arcs": [], "lwpolylines": [], "polylines": [], "solids": [],
                "ellipses": [], "splines": []}
    for entity in cluster:
        transform_matrix = transform_matrices[entity]

        entity_color = get_entity_color(entity, layer_properties, header_defaults, metadata["background_color"])
        line_weight = get_entity_lineweight(entity, layer_properties, header_defaults)
        line_style = get_entity_linetype(entity, layer_properties, header_defaults)
        layer = get_entity_layer(entity, layer_properties, header_defaults)
        if entity.dxftype() == 'LINE':
            start = format_point2(entity_to_points[entity][0])
            end = format_point2(entity_to_points[entity][1])
            contours["lines"].append(
                {"start": start, "end": end, "colour": entity_color, "weight": line_weight,
                 "style": line_style, "layer": layer})
        elif entity.dxftype() == 'CIRCLE':
            center = format_point2(transform_point_to_tuple(entity.dxf.center.x if entity.dxf.center.x > 0 else -entity.dxf.center.x, entity.dxf.center.y, transform_matrix))
            contours["circles"].append(
                {"centre": center, "radius": transform_height(entity.dxf.radius, transform_matrix), "colour": entity_color, "weight": line_weight,
                 "style": line_style, "layer": layer})
        elif entity.dxftype() == 'ARC':
            center = format_point2(transform_point_to_tuple(entity.dxf.center.x, entity.dxf.center.y, transform_matrix))
            contours["arcs"].append(
                {"centre": center, "radius": transform_height(entity.dxf.radius, transform_matrix), "start_angle": entity.dxf.start_angle,
                 "end_angle": entity.dxf.end_angle, "colour": entity_color, "weight": line_weight, "style": line_style,
                 "layer": layer})
        elif entity.dxftype() == 'LWPOLYLINE':
            #points = [format_point2(transform_point(p[0], p[1], transform_matrix)) for p in entity.get_points()]
            points = [format_point2(p) for p in entity_to_points[entity]]
            contours["lwpolylines"].append(
                {"points": points, "colour": entity_color, "weight": line_weight, "style": line_style, "layer": layer})
        elif entity.dxftype() == 'POLYLINE':
            #points = [format_point2(transform_point(v.dxf.location.x, v.dxf.location.y, transform_matrix)) for v in
            #          entity.vertices]
            points = [format_point2(p) for p in entity_to_points[entity]]
            contours["polylines"].append(
                {"points": points, "colour": entity_color, "weight": line_weight, "style": line_style, "layer": layer})
        elif entity.dxftype() == 'SOLID':
            points = [format_point2(p) for p in entity_to_points[entity]]
            contours["solids"].append(
                {"points": points, "colour": entity_color, "weight": line_weight, "style": line_style, "layer": layer})
        elif entity.dxftype() == 'ELLIPSE':
            center = format_point2(transform_point_to_tuple(entity.dxf.center.x, entity.dxf.center.y, transform_matrix))
            major_axis_vector = transform_point_to_tuple(entity.dxf.major_axis[0], entity.dxf.major_axis[1], transform_matrix)
            major_axis_length = np.linalg.norm([major_axis_vector[0], major_axis_vector[1]])
            minor_axis_length = major_axis_length * entity.dxf.ratio
            rotation_angle = np.arctan2(major_axis_vector[1], major_axis_vector[0])
            contours["ellipses"].append({"centre": center, "major_axis_length": major_axis_length * 2,
                                         "minor_axis_length": minor_axis_length * 2,
                                         "rotation_angle": np.degrees(rotation_angle), "colour": entity_color,
                                         "weight": line_weight, "style": line_style, "layer": layer})
        elif entity.dxftype() == 'SPLINE':
            points = [format_point2(p) for p in entity_to_points[entity]]
            contours["splines"].append(
                {"points": points, "colour": entity_color, "weight": line_weight, "style": line_style, "layer": layer})
    return contours


def classify_text_entities(all_entities, text_styles, metadata, layer_properties, header_defaults):
    texts = {'texts': [], 'mtexts': [], 'attdefs': []}
    for entity in all_entities:
        if entity.dxftype() != 'TEXT' and entity.dxftype() != 'MTEXT':
            continue
        entity_color = get_entity_color(entity, layer_properties, header_defaults, metadata["background_color"])
        line_weight = get_entity_lineweight(entity, layer_properties, header_defaults)
        line_style = get_entity_linetype(entity, layer_properties, header_defaults)
        layer = get_entity_layer(entity, layer_properties, header_defaults)
        if entity.dxftype() == 'TEXT':
            text_center = transform_point_to_tuple(entity.dxf.insert.x, entity.dxf.insert.y, np.identity(3))
            text_height = transform_height(entity.dxf.height, np.identity(3))
            font = get_font_for_style(entity.dxf.style, text_styles)
            text_data = {
                "text": entity.dxf.text,
                "center": text_center,
                "height": text_height,
                "style": entity.dxf.style,
                "font": font,
                "color": "#000000"
            }

            texts["texts"].append(text_data)
        elif entity.dxftype() == 'MTEXT':
            text = re.sub(r'\\f[^;]*;|\\[A-Za-z]+\;|\\H\d+\.\d+;|\\P|{\\H[^}]*;|}|{|}|\\W\d+\.\d+;|\\pxa\d+\.\d+,t\d+;', '', entity.text)
            text_center = transform_point_to_tuple(entity.dxf.insert.x, entity.dxf.insert.y, np.identity(3))
            text_height = transform_height(entity.dxf.char_height, np.identity(3))
            text_direction = list(entity.dxf.text_direction)
            font = get_font_for_style(entity.dxf.style, text_styles)
            text_data = {
                "text": text,
                "center": text_center,
                "attachment_point": entity.dxf.attachment_point,
                "text_direction": text_direction,
                "height": text_height,
                "style": entity.dxf.style,
                "font": font,
                "color": "#000000"
            }
            texts["mtexts"].append(text_data)
    return texts


def process_border_block(border_inserts, doc_blocks, metadata, layer_properties, header_defaults):
    border_contours = {
        "lines": [], "circles": [], "arcs": [], "lwpolylines": [], "polylines": [], "solids": [], "ellipses": [],
        "splines": []
    }
    all_entities = []
    all_entity_to_points = {}
    all_transform_matrices = {}

    for be, matrix in border_inserts:
        block = doc_blocks.get(be.dxf.name)
        _, block_entity_to_points, block_transform_matrices, *_ = process_entities(doc_blocks, list(block), None, matrix)
        all_entities.extend(block_entity_to_points.keys())
        all_entity_to_points.update(block_entity_to_points)
        all_transform_matrices.update(block_transform_matrices)

    classified = classify_contour_entities(all_entities, all_transform_matrices, all_entity_to_points, metadata, layer_properties, header_defaults)
    for key in border_contours.keys():
        border_contours[key].extend(classified.get(key, []))

    border_view = {"contours": border_contours, "block_name": "Border"}

    return border_view


def process_dimension_geometries(dimension_entities, doc_blocks, metadata, layer_properties, header_defaults, alpha_shapes, text_styles):
    dimension_geometries = {}

    for de in dimension_entities:
        d_block_pointer = de.dxf.get('geometry', None)
        if d_block_pointer:
            d_block = doc_blocks[d_block_pointer]
            dimension_geometry = {}
            d_geometry_contours = {
                "lines": [], "circles": [], "arcs": [], "lwpolylines": [], "polylines": [], "solids": [], "ellipses": [],
                "splines": []
            }
            all_d_block_entities = []
            all_d_block_entities_to_points = {}
            all_d_block_entity_transform_matrices = {}

            _, d_block_entities_to_points, d_block_transform_matrices, *_ = process_entities(doc_blocks, list(d_block), None)  # matrix missing
            all_d_block_entities.extend(d_block_entities_to_points.keys())
            all_d_block_entities_to_points.update(d_block_entities_to_points)
            all_d_block_entity_transform_matrices.update(d_block_transform_matrices)
            d_block_texts = classify_text_entities(all_d_block_entities, text_styles, metadata, layer_properties, header_defaults)

            classified = classify_contour_entities(all_d_block_entities, all_d_block_entity_transform_matrices,
                                                   all_d_block_entities_to_points, metadata, layer_properties,
                                                   header_defaults)
            for key in d_geometry_contours.keys():
                d_geometry_contours[key].extend(classified.get(key, []))
            dimension_geometry["contours"] = d_geometry_contours
            dimension_geometry["view"] = find_closest_view(de.dxf.get('defpoint', None), alpha_shapes)
            dimension_geometry["texts"] = d_block_texts
            dimension_geometries[d_block_pointer] = dimension_geometry

    return dimension_geometries


def find_closest_view(text_center, alpha_shapes):
    min_distance = float('inf')
    closest_view = None
    point = Point(text_center)
    for view, alpha_shape_coords in alpha_shapes.items():
        # alpha_shape = Polygon(alpha_shape_coords)
        # dist = point.distance(alpha_shape)
        dist = point.distance(alpha_shape_coords)
        if dist < min_distance:
            min_distance = dist
            closest_view = view
    return closest_view
