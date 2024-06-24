import ezdxf
import numpy as np
from collections import defaultdict
from app.parsers.utilities import normalize_point2, transform_point, apply_transform
from app.parsers.parsing_utilities import get_entity_color, get_entity_lineweight, get_entity_linetype, get_entity_layer, get_insert_transform
import re


def process_entities(doc, entities, metadata, parent_transform=np.identity(3)):
    points = []
    entity_to_points = defaultdict(list)
    transform_matrices = {}
    border_entities = []
    dimensions = []
    texts = []

    def process_block(block, transform_matrix):
        block_points = []
        block_entity_to_points = defaultdict(list)
        block_transform_matrices = {}
        block_texts = []
        for entity in block:
            if entity.dxftype() == 'INSERT':
                insert_matrix = get_insert_transform(entity)
                combined_matrix = np.dot(transform_matrix, insert_matrix)
                nested_block = doc.blocks.get(entity.dxf.name)
                if 'border' in entity.dxf.name.lower():
                    border_entities.append((entity, combined_matrix))
                    transform_matrices[entity] = combined_matrix
                    continue
                nested_points, nested_entity_to_points, nested_transform_matrices = process_block(nested_block, combined_matrix)
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
                if entity_points:
                    transformed_points = apply_transform(transform_matrix, entity_points)
                    block_entity_to_points[entity].extend(transformed_points)
                    block_transform_matrices[entity] = transform_matrix
                    block_points.extend(transformed_points)
        return block_points, block_entity_to_points, block_transform_matrices, block_texts

    for entity in entities:
        if entity.dxftype() == 'INSERT':
            insert_matrix = get_insert_transform(entity)
            block = doc.blocks.get(entity.dxf.name)
            if 'border' in entity.dxf.name.lower():
                border_entities.append((entity, insert_matrix))
                transform_matrices[entity] = insert_matrix
                continue
            block_points, block_entity_to_points, block_transform_matrices, block_texts = process_block(block, insert_matrix)
            points.extend(block_points)
            for k, v in block_entity_to_points.items():
                entity_to_points[k].extend(v)
            transform_matrices.update(block_transform_matrices)
        elif entity.dxftype() == 'DIMENSION':
            dimensions.append(entity)
        else:
            entity_points = extract_points_from_entity(entity)
            if entity_points:
                transformed_points = apply_transform(parent_transform, entity_points)
                entity_to_points[entity].extend(transformed_points)
                transform_matrices[entity] = parent_transform
                points.extend(transformed_points)

    return points, entity_to_points, transform_matrices, border_entities, dimensions


def extract_points_from_entity(entity):
    num_segments = 72
    if entity.dxftype() == 'LINE':
        return [np.array(entity.dxf.start), np.array(entity.dxf.end)]
    elif entity.dxftype() == 'CIRCLE':
        center = np.array(entity.dxf.center)
        radius = entity.dxf.radius
        angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        return [center + radius * np.array([np.cos(a), np.sin(a), 0]) for a in angles]
    elif entity.dxftype() == 'ARC':
        center = np.array(entity.dxf.center)
        radius = entity.dxf.radius
        start_angle = np.radians(entity.dxf.start_angle)
        end_angle = np.radians(entity.dxf.end_angle)
        angles = np.linspace(start_angle, end_angle, num_segments, endpoint=True)
        return [center + radius * np.array([np.cos(a), np.sin(a), 0]) for a in angles]
    elif entity.dxftype() == 'ELLIPSE':
        center = np.array(entity.dxf.center)
        major_axis = np.array([entity.dxf.major_axis.x, entity.dxf.major_axis.y])
        major_axis_length = np.linalg.norm(major_axis)
        minor_axis_length = major_axis_length * entity.dxf.ratio
        rotation_angle = np.arctan2(major_axis[1], major_axis[0])
        angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        return [center + np.array([major_axis_length * np.cos(a) * np.cos(rotation_angle) - minor_axis_length * np.sin(
            a) * np.sin(rotation_angle),
                                   major_axis_length * np.cos(a) * np.sin(rotation_angle) + minor_axis_length * np.sin(
                                       a) * np.cos(rotation_angle), 0]) for a in angles]
    elif entity.dxftype() == 'SPLINE':
        return [np.array(point) for point in entity.fit_points]
    elif entity.dxftype() == 'POLYLINE':
        return [np.array(vertex.dxf.location) for vertex in entity.vertices]
    elif entity.dxftype() == 'HATCH':
        points = []
        for path in entity.paths:
            if isinstance(path, ezdxf.entities.PolylinePath):
                points.extend([np.array((v.x, v.y, 0)) for v in path.vertices])
            elif isinstance(path, ezdxf.entities.EdgePath):
                for edge in path.edges:
                    if edge.EDGE_TYPE == 'LineEdge':
                        points.extend(
                            [np.array((edge.start.x, edge.start.y, 0)), np.array((edge.end.x, edge.end.y, 0))])
                    elif edge.EDGE_TYPE == 'ArcEdge':
                        center = np.array(edge.center)
                        radius = edge.radius
                        start_angle = np.radians(edge.start_angle)
                        end_angle = np.radians(edge.end_angle)
                        angles = np.linspace(start_angle, end_angle, num_segments, endpoint=True)
                        points.extend([center + radius * np.array([np.cos(a), np.sin(a), 0]) for a in angles])
        return points
    return []


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
            start = normalize_point2(transform_point(entity.dxf.start.x, entity.dxf.start.y, transform_matrix))
            end = normalize_point2(transform_point(entity.dxf.end.x, entity.dxf.end.y, transform_matrix))
            contours["lines"].append(
                {"start": start, "end": end, "colour": entity_color, "weight": line_weight, "style": line_style, "layer": layer})
        elif entity.dxftype() == 'CIRCLE':
            center = normalize_point2(transform_point(entity.dxf.center.x, entity.dxf.center.y, transform_matrix))
            contours["circles"].append(
                {"centre": center, "radius": entity.dxf.radius, "colour": entity_color, "weight": line_weight,
                 "style": line_style, "layer": layer})
        elif entity.dxftype() == 'ARC':
            center = normalize_point2(transform_point(entity.dxf.center.x, entity.dxf.center.y, transform_matrix))
            contours["arcs"].append(
                {"centre": center, "radius": entity.dxf.radius, "start_angle": entity.dxf.start_angle,
                 "end_angle": entity.dxf.end_angle, "colour": entity_color, "weight": line_weight, "style": line_style,
                 "layer": layer})
        elif entity.dxftype() == 'LWPOLYLINE':
            points = [normalize_point2(transform_point(p[0], p[1], transform_matrix)) for p in entity.get_points()]
            contours["lwpolylines"].append(
                {"points": points, "colour": entity_color, "weight": line_weight, "style": line_style, "layer": layer})
        elif entity.dxftype() == 'POLYLINE':
            points = [normalize_point2(transform_point(v.dxf.location.x, v.dxf.location.y, transform_matrix)) for v in
                      entity.vertices]
            contours["polylines"].append(
                {"points": points, "colour": entity_color, "weight": line_weight, "style": line_style, "layer": layer})
        elif entity.dxftype() == 'SOLID':
            points = [normalize_point2(transform_point(entity.dxf.vtx0.x, entity.dxf.vtx0.y, transform_matrix)),
                      normalize_point2(transform_point(entity.dxf.vtx1.x, entity.dxf.vtx1.y, transform_matrix)),
                      normalize_point2(transform_point(entity.dxf.vtx2.x, entity.dxf.vtx2.y, transform_matrix)),
                      normalize_point2(transform_point(entity.dxf.vtx3.x, entity.dxf.vtx3.y, transform_matrix))]
            contours["solids"].append(
                {"points": points, "colour": entity_color, "weight": line_weight, "style": line_style, "layer": layer})
        elif entity.dxftype() == 'ELLIPSE':
            center = normalize_point2(transform_point(entity.dxf.center.x, entity.dxf.center.y, transform_matrix))
            major_axis_vector = transform_point(entity.dxf.major_axis[0], entity.dxf.major_axis[1], transform_matrix)
            major_axis_length = np.linalg.norm([major_axis_vector[0], major_axis_vector[1]])
            minor_axis_length = major_axis_length * entity.dxf.ratio
            rotation_angle = np.arctan2(major_axis_vector[1], major_axis_vector[0])
            contours["ellipses"].append({"centre": center, "major_axis_length": major_axis_length * 2,
                                         "minor_axis_length": minor_axis_length * 2,
                                         "rotation_angle": np.degrees(rotation_angle), "colour": entity_color,
                                         "weight": line_weight, "style": line_style, "layer": layer})
        elif entity.dxftype() == 'SPLINE':
            points = [normalize_point2(transform_point(p[0], p[1], transform_matrix)) for p in entity.fit_points]
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
            texts["texts"].append({"text": text, "height": height, "style": style, "colour": entity_color, "layer": layer})
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
            texts["mtexts"].append({"text": text, "center": text_center, "height": height, "style": style, "colour": "#000000", "layer": layer})
    return texts
