import ezdxf
import numpy as np
from collections import defaultdict
from shapely.geometry import MultiPoint
import alphashape
from shapely.strtree import STRtree
from scipy.spatial import cKDTree, KDTree
from app.parsers.utilities import normalize_point2, map_color, transform_point


def get_insert_transform(insert):
    m = np.identity(3)
    scale_x, scale_y = insert.dxf.xscale, insert.dxf.yscale
    angle = np.deg2rad(insert.dxf.rotation)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # Apply scaling
    m[0, 0] *= scale_x
    m[1, 1] *= scale_y
    # Apply rotation
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    m = np.dot(m, rotation_matrix)
    # Apply translation
    m[0, 2] += insert.dxf.insert.x
    m[1, 2] += insert.dxf.insert.y
    return m


def transform_point(x, y, matrix):
    point = np.dot(matrix, [x, y, 1])
    return point[0], point[1]


def get_entity_color(entity, layer_properties, header_defaults, background_color):
    if entity.dxf.hasattr('color') and entity.dxf.color != 256:  # 256 indicates "by layer"
        return map_color(entity.dxf.color, background_color)
    else:
        layer_color = layer_properties[entity.dxf.layer]["color"] if entity.dxf.layer in layer_properties else \
            header_defaults["color"]
        return map_color(layer_color, background_color)


def get_entity_lineweight(entity, layer_properties, header_defaults):
    if entity.dxf.hasattr('lineweight') and entity.dxf.lineweight != -1:  # -1 indicates "by layer"
        return entity.dxf.lineweight / 100.0
    else:
        layer_lineweight = layer_properties[entity.dxf.layer]["lineweight"] if entity.dxf.layer in layer_properties else \
            header_defaults["lineweight"]
        return layer_lineweight / 100.0 if layer_lineweight > 0 else header_defaults["lineweight"] / 100.0


def get_entity_linetype(entity, layer_properties, header_defaults):
    if entity.dxf.hasattr('linetype') and entity.dxf.linetype.lower() != "bylayer":
        return entity.dxf.linetype
    else:
        return layer_properties[entity.dxf.layer]["linetype"] if entity.dxf.layer in layer_properties else \
            header_defaults["linetype"]


def get_entity_layer(entity, layer_properties, header_defaults):
    return entity.dxf.get("layer", "unknown")


def map_color(color, background_color):
    # Map AutoCAD color index to actual hex color code
    color_mapping = {
        1: '0xFF0000',  # Red
        2: '0xFFFF00',  # Yellow
        3: '0x00FF00',  # Green
        4: '0x00FFFF',  # Cyan
        5: '0x0000FF',  # Blue
        6: '0xFF00FF',  # Magenta
        7: '0xFFFFFF' if background_color.lower() == "0x000000" else '0x000000',  # White/Black
        8: '0x808080',  # Gray
        9: '0xC0C0C0',  # Light Gray
        # Add more color mappings as needed
    }
    return color_mapping.get(color, '0x000000')  # Default to black if color not found


def process_entities(doc, entities, metadata, parent_transform=np.identity(3)):
    points = []
    entity_to_points = defaultdict(list)
    transform_matrices = {}
    border_entities = []
    dimensions = []

    def apply_transform(matrix, points):
        return [transform_point(p[0], p[1], matrix) for p in points]

    def process_block(block, transform_matrix):
        block_points = []
        block_entity_to_points = defaultdict(list)
        block_transform_matrices = {}
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
            else:
                entity_points = extract_points_from_entity(entity)
                if entity_points:
                    transformed_points = apply_transform(transform_matrix, entity_points)
                    block_entity_to_points[entity].extend(transformed_points)
                    block_transform_matrices[entity] = transform_matrix
                    block_points.extend(transformed_points)
        return block_points, block_entity_to_points, block_transform_matrices

    for entity in entities:
        if entity.dxftype() == 'INSERT':
            insert_matrix = get_insert_transform(entity)
            block = doc.blocks.get(entity.dxf.name)
            if 'border' in entity.dxf.name.lower():
                border_entities.append((entity, insert_matrix))
                transform_matrices[entity] = insert_matrix
                continue
            block_points, block_entity_to_points, block_transform_matrices = process_block(block, insert_matrix)
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
                new_cluster = clusters[idx1] + clusters[idx2]
                merged.add(idx1)
                merged.add(idx2)

                # Assign a new index for the new cluster
                new_idx = len(new_alpha_shapes)
                new_alpha_shapes[new_idx] = get_alpha_shape(new_cluster, alpha)
                cluster_mapping[new_idx] = new_cluster
                merged_current = True
                #print(f"Merged clusters {idx1} and {idx2} into new cluster {new_idx}.")
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
        print(f"Iteration {iterations}: {len(clusters)} clusters before merge.")
        num_clusters_before = len(clusters)
        clusters, alpha_shapes = merge_clusters_with_alpha_shape(clusters, alpha, alpha_shapes)
        num_clusters_after = len(clusters)

        if num_clusters_before == num_clusters_after:
            break
        iterations += 1

    return clusters


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
            text = entity.text
            height = entity.dxf.char_height
            style = entity.dxf.style
            texts["mtexts"].append({"text": text, "height": height, "style": style, "colour": entity_color, "layer": layer})
    return texts

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
            points.extend([center + np.array([major_axis_length * np.cos(a) * np.cos(
                rotation_angle) - minor_axis_length * np.sin(a) * np.sin(rotation_angle),
                                              major_axis_length * np.cos(a) * np.sin(
                                                  rotation_angle) + minor_axis_length * np.sin(a) * np.cos(
                                                  rotation_angle), 0]) for a in angles])
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
