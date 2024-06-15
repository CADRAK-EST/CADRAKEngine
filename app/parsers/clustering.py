﻿import ezdxf
import numpy as np
from collections import defaultdict
from shapely.geometry import MultiPoint
import alphashape
from shapely.strtree import STRtree
from app.parsers.utilities import normalize_point2

def map_color(color, background_color):
    color_mapping = {
        1: '0xFF0000', 2: '0xFFFF00', 3: '0x00FF00', 4: '0x00FFFF', 5: '0x0000FF', 6: '0xFF00FF',
        7: '0xFFFFFF' if background_color.lower() == "0x000000" else '0x000000', 8: '0x808080', 9: '0xC0C0C0'
    }
    return color_mapping.get(color, '0x000000')

def transform_point(x, y, matrix):
    point = np.dot(matrix, [x, y, 1])
    return point[0], point[1]

def process_entities(doc, entities, metadata):
    points = []
    entity_to_points = defaultdict(list)
    transform_matrices = {}

    def get_insert_transform(insert):
        m = np.identity(3)
        scale_x, scale_y = insert.dxf.xscale, insert.dxf.yscale
        angle = np.deg2rad(insert.dxf.rotation)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        m[0, 0] *= scale_x
        m[1, 1] *= scale_y
        rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        m = np.dot(m, rotation_matrix)
        m[0, 2] += insert.dxf.insert.x
        m[1, 2] += insert.dxf.insert.y
        return m

    for entity in entities:
        if entity.dxftype() == 'INSERT' and not entity.dxf.name.startswith('Border'):
            block = doc.blocks.get(entity.dxf.name)
            insert_matrix = get_insert_transform(entity)
            combined_matrix = np.dot(np.identity(3), insert_matrix)
            block_points, block_entity_to_points, block_transform_matrices = process_entities(doc, block, metadata)
            points.extend(block_points)
            entity_to_points.update(block_entity_to_points)
            transform_matrices.update(block_transform_matrices)
        else:
            entity_points = extract_points_from_entity(entity)
            if entity_points:
                transform_matrix = np.identity(3)  # Apply any necessary transformation here
                transformed_points = [transform_point(p[0], p[1], transform_matrix) for p in entity_points]
                entity_to_points[entity] = transformed_points
                transform_matrices[entity] = transform_matrix
                points.extend(transformed_points)

    return points, entity_to_points, transform_matrices

def extract_points_from_entity(entity):
    num_segments = 72
    if entity.dxftype() == 'LINE':
        return [entity.dxf.start, entity.dxf.end]
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
        return [center + np.array([major_axis_length * np.cos(a) * np.cos(rotation_angle) - minor_axis_length * np.sin(a) * np.sin(rotation_angle),
                                   major_axis_length * np.cos(a) * np.sin(rotation_angle) + minor_axis_length * np.sin(a) * np.cos(rotation_angle), 0]) for a in angles]
    elif entity.dxftype() == 'SPLINE':
        return entity.fit_points
    elif entity.dxftype() == 'POLYLINE':
        return [vertex.dxf.location for vertex in entity.vertices]
    elif entity.dxftype() == 'HATCH':
        points = []
        for path in entity.paths:
            if isinstance(path, ezdxf.entities.PolylinePath):
                points.extend([(v.x, v.y, 0) for v in path.vertices])
            elif isinstance(path, ezdxf.entities.EdgePath):
                for edge in path.edges:
                    if edge.EDGE_TYPE == 'LineEdge':
                        points.extend([(edge.start.x, edge.start.y, 0), (edge.end.x, edge.end.y, 0)])
                    elif edge.EDGE_TYPE == 'ArcEdge':
                        center = np.array(edge.center)
                        radius = edge.radius
                        start_angle = np.radians(edge.start_angle)
                        end_angle = np.radians(edge.end_angle)
                        angles = np.linspace(start_angle, end_angle, num_segments, endpoint=True)
                        points.extend([center + radius * np.array([np.cos(a), np.sin(a), 0]) for a in angles])
        return points
    return []

def iterative_merge(clusters, alpha):
    while True:
        new_clusters = merge_clusters_with_alpha_shape(clusters, alpha)
        if len(new_clusters) == len(clusters):
            break
        clusters = new_clusters
    return clusters

def merge_clusters_with_alpha_shape(clusters, alpha):
    new_clusters = []
    merged = set()
    alpha_shapes = [get_alpha_shape(cluster, alpha) for cluster in clusters]
    spatial_index = STRtree(alpha_shapes)

    for i, alpha_shape1 in enumerate(alpha_shapes):
        if i in merged:
            continue

        merged_current = False
        for j in spatial_index.query(alpha_shape1):
            if i >= j or j in merged:
                continue

            alpha_shape2 = alpha_shapes[j]

            if alpha_shape1.intersects(alpha_shape2):
                new_clusters.append(clusters[i] + clusters[j])
                merged.add(i)
                merged.add(j)
                merged_current = True
                break

        if not merged_current:
            new_clusters.append(clusters[i])

    return new_clusters

def classify_entities(cluster, transform_matrices, metadata):
    contours = {"lines": [], "circles": [], "arcs": [], "lwpolylines": [], "polylines": [], "solids": []}
    for entity in cluster:
        transform_matrix = transform_matrices[entity]
        entity_color = map_color(entity.dxf.color, metadata["background_color"]) if entity.dxf.hasattr('color') else '0x000000'
        line_weight = entity.dxf.lineweight / 100.0 if entity.dxf.hasattr('lineweight') else -1
        line_style = entity.dxf.linetype if entity.dxf.hasattr('linetype') else 'BYLAYER'
        if entity.dxftype() == 'LINE':
            start = normalize_point2(transform_point(entity.dxf.start.x, entity.dxf.start.y, transform_matrix))
            end = normalize_point2(transform_point(entity.dxf.end.x, entity.dxf.end.y, transform_matrix))
            contours["lines"].append({"start": start, "end": end, "colour": entity_color, "weight": line_weight, "style": line_style})
        elif entity.dxftype() == 'CIRCLE':
            center = normalize_point2(transform_point(entity.dxf.center.x, entity.dxf.center.y, transform_matrix))
            contours["circles"].append({"centre": center, "radius": entity.dxf.radius, "colour": entity_color, "weight": line_weight, "style": line_style})
        elif entity.dxftype() == 'ARC':
            center = normalize_point2(transform_point(entity.dxf.center.x, entity.dxf.center.y, transform_matrix))
            contours["arcs"].append({"centre": center, "radius": entity.dxf.radius, "start_angle": entity.dxf.start_angle, "end_angle": entity.dxf.end_angle, "colour": entity_color, "weight": line_weight, "style": line_style})
        elif entity.dxftype() == 'LWPOLYLINE':
            points = [normalize_point2(transform_point(p[0], p[1], transform_matrix)) for p in entity.get_points()]
            contours["lwpolylines"].append({"points": points, "colour": entity_color, "weight": line_weight, "style": line_style})
        elif entity.dxftype() == 'POLYLINE':
            points = [normalize_point2(transform_point(v.dxf.location.x, v.dxf.location.y, transform_matrix)) for v in entity.vertices]
            contours["polylines"].append({"points": points, "colour": entity_color, "weight": line_weight, "style": line_style})
        elif entity.dxftype() == 'SOLID':
            points = [normalize_point2(transform_point(entity.dxf.vtx0.x, entity.dxf.vtx0.y, transform_matrix)), normalize_point2(transform_point(entity.dxf.vtx1.x, entity.dxf.vtx1.y, transform_matrix)), normalize_point2(transform_point(entity.dxf.vtx2.x, entity.dxf.vtx2.y, transform_matrix)), normalize_point2(transform_point(entity.dxf.vtx3.x, entity.dxf.vtx3.y, transform_matrix))]
            contours["solids"].append({"points": points, "colour": entity_color, "weight": line_weight, "style": line_style})
    return contours

def get_alpha_shape(cluster, alpha=0.1):
    points = entities_to_points(cluster)
    if len(points) < 4:
        return MultiPoint(points).convex_hull
    try:
        return alphashape.alphashape(points, alpha)
    except Exception:
        return MultiPoint(points).convex_hull

def entities_to_points(cluster):
    points = []
    num_segments = 72
    for entity in cluster:
        if entity.dxftype() == 'LINE':
            points.extend([entity.dxf.start, entity.dxf.end])
        elif entity.dxftype() == 'CIRCLE':
            center = np.array(entity.dxf.center)
            radius = entity.dxf.radius
            angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
            points.extend([center + radius * np.array([np.cos(a), np.sin(a), 0]) for a in angles])
        elif entity.dxftype() == 'ARC':
            center = np.array(entity.dxf.center)
            radius = entity.dxf.radius
            start_angle = np.radians(entity.dxf.start_angle)
            end_angle = np.radians(entity.dxf.end_angle)
            angles = np.linspace(start_angle, end_angle, num_segments, endpoint=True)
            points.extend([center + radius * np.array([np.cos(a), np.sin(a), 0]) for a in angles])
        elif entity.dxftype() == 'ELLIPSE':
            center = np.array(entity.dxf.center)
            major_axis = np.array([entity.dxf.major_axis.x, entity.dxf.major_axis.y])
            major_axis_length = np.linalg.norm(major_axis)
            minor_axis_length = major_axis_length * entity.dxf.ratio
            rotation_angle = np.arctan2(major_axis[1], major_axis[0])
            angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
            points.extend([center + np.array([major_axis_length * np.cos(a) * np.cos(rotation_angle) - minor_axis_length * np.sin(a) * np.sin(rotation_angle),
                                              major_axis_length * np.cos(a) * np.sin(rotation_angle) + minor_axis_length * np.sin(a) * np.cos(rotation_angle), 0]) for a in angles])
        elif entity.dxftype() == 'SPLINE':
            points.extend(entity.fit_points)
        elif entity.dxftype() == 'POLYLINE':
            points.extend([vertex.dxf.location for vertex in entity.vertices])
        elif entity.dxftype() == 'HATCH':
            for path in entity.paths:
                if isinstance(path, ezdxf.entities.PolylinePath):
                    points.extend([(v.x, v.y, 0) for v in path.vertices])
                elif isinstance(path, ezdxf.entities.EdgePath):
                    for edge in path.edges:
                        if edge.EDGE_TYPE == 'LineEdge':
                            points.extend([(edge.start.x, edge.start.y, 0), (edge.end.x, edge.end.y, 0)])
                        elif edge.EDGE_TYPE == 'ArcEdge':
                            center = np.array(edge.center)
                            radius = edge.radius
                            start_angle = np.radians(edge.start_angle)
                            end_angle = np.radians(edge.end_angle)
                            angles = np.linspace(start_angle, end_angle, num_segments, endpoint=True)
                            points.extend([center + radius * np.array([np.cos(a), np.sin(a), 0]) for a in angles])
    return points
