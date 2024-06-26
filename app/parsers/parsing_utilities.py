import numpy as np


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
        layer_lineweight = layer_properties[entity.dxf.layer]["lineweight"] if entity.dxf.layer in layer_properties\
            else header_defaults["lineweight"]
        return layer_lineweight / 100.0 if layer_lineweight > 0 else header_defaults["lineweight"] / 100.0


def get_entity_linetype(entity, layer_properties, header_defaults):
    if entity.dxf.hasattr('linetype') and entity.dxf.linetype.lower() != "bylayer":
        return entity.dxf.linetype
    else:
        return layer_properties[entity.dxf.layer]["linetype"] if entity.dxf.layer in layer_properties else \
            header_defaults["linetype"]


def get_entity_layer(entity, layer_properties, header_defaults):
    return entity.dxf.get("layer", "unknown")

