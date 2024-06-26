import ezdxf
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
import numpy as np


def list_blocks_and_visualize(file_path):
    # Load the DXF document
    doc = ezdxf.readfile(file_path)

    # List all blocks and their entities, and visualize them
    for block in doc.blocks:
        print(f"Block Name: {block.name}")
        print("Entities:")

        # Close the previous plot if any
        plt.close('all')

        fig, ax = plt.subplots()

        legend_entries = set()
        for entity in block:
            entity_type = entity.dxftype()
            try:
                layer = entity.dxf.layer

                print(f"  - Type: {entity_type}, Layer: {layer}, Attributes: {entity.dxf.all_existing_dxf_attribs()}")

                # Visualization
                if entity_type == 'LINE':
                    start = entity.dxf.start
                    end = entity.dxf.end
                    ax.plot([start.x, end.x], [start.y, end.y], label=f"{entity_type}" if entity_type not in legend_entries else "")
                    legend_entries.add(entity_type)

                elif entity_type == 'CIRCLE':
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    circle = Circle((center.x, center.y), radius, fill=False, label=f"{entity_type}" if entity_type not in legend_entries else "")
                    ax.add_patch(circle)
                    legend_entries.add(entity_type)

                elif entity_type == 'ARC':
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    start_angle = entity.dxf.start_angle
                    end_angle = entity.dxf.end_angle
                    arc = Arc((center.x, center.y), radius*2, radius*2, angle=0, theta1=start_angle, theta2=end_angle, label=f"{entity_type}" if entity_type not in legend_entries else "")
                    ax.add_patch(arc)
                    legend_entries.add(entity_type)

                elif entity_type == 'POLYLINE':
                    points = np.array(entity.points())
                    if points.ndim == 2 and points.shape[0] > 0:
                        ax.plot(points[:, 0], points[:, 1], label=f"{entity_type}" if entity_type not in legend_entries else "")
                        legend_entries.add(entity_type)

                elif entity_type == 'LWPOLYLINE':
                    points = np.array(list(entity.get_points()))
                    if points.ndim == 2 and points.shape[0] > 0:
                        ax.plot(points[:, 0], points[:, 1], label=f"{entity_type}" if entity_type not in legend_entries else "")
                        legend_entries.add(entity_type)

                elif entity_type == 'SPLINE':
                    fit_points = np.array(entity.fit_points)
                    if fit_points.ndim == 2 and fit_points.shape[0] > 0:
                        ax.plot(fit_points[:, 0], fit_points[:, 1], label=f"{entity_type}" if entity_type not in legend_entries else "")
                        legend_entries.add(entity_type)

            except AttributeError:
                print(f"  - Type: {entity_type} (no layer attribute)")

        ax.set_aspect('equal')
        ax.autoscale_view()
        plt.title(f'Block: {block.name}')

        # Add legend only if there are entries
        if legend_entries:
            plt.legend()

        plt.show()

# Example usage
file_path = "C:\\Users\\marat\\Documents\\Github\\CADRAK2\\CADRAKEngine\\test_data\\Loik1.dxf"
list_blocks_and_visualize(file_path)
"""

import ezdxf
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
import numpy as np

def parse_dxf_by_layers(file_path):
    # Load the DXF document
    doc = ezdxf.readfile(file_path)

    # Create a dictionary to hold entities by layer
    entities_by_layer = {}

    # Iterate over entities and group them by layer
    for entity in doc.entities:
        entity_type = entity.dxftype()
        try:
            layer = entity.dxf.layer
        except ezdxf.lldxf.const.DXFAttributeError:
            layer = "No Layer"  # Fallback if the entity does not have a layer attribute

        if layer not in entities_by_layer:
            entities_by_layer[layer] = []
        entities_by_layer[layer].append(entity)

    # Print entities grouped by layer
    for layer, entities in entities_by_layer.items():
        print(f"Layer: {layer}")
        for entity in entities:
            print(f"  - Type: {entity.dxftype()}, Attributes: {entity.dxf.all_existing_dxf_attribs()}")

def parse_dxf_blocks_and_inserts(file_path):
    # Load the DXF document
    doc = ezdxf.readfile(file_path)

    # Create a dictionary to hold blocks and their inserts
    inserts_by_block = {}

    # Iterate over all entities and collect INSERT entities
    for entity in doc.entities:
        if entity.dxftype() == 'INSERT':
            block_name = entity.dxf.name
            if block_name not in inserts_by_block:
                inserts_by_block[block_name] = []
            inserts_by_block[block_name].append(entity)

    # Print block references and their attributes
    for block_name, inserts in inserts_by_block.items():
        print(f"Block: {block_name}")
        for insert in inserts:
            print(f"  - Insert: {insert.dxf.handle}, Attributes: {insert.dxf.all_existing_dxf_attribs()}")

def parse_dxf_by_entity_types(file_path):
    # Load the DXF document
    doc = ezdxf.readfile(file_path)

    # Create a dictionary to hold entities by type
    entities_by_type = {}

    # Iterate over entities and group them by type
    for entity in doc.entities:
        entity_type = entity.dxftype()
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []
        entities_by_type[entity_type].append(entity)

    # Print entities grouped by type
    for entity_type, entities in entities_by_type.items():
        print(f"Entity Type: {entity_type}")
        for entity in entities:
            print(f"  - Attributes: {entity.dxf.all_existing_dxf_attribs()}")

# Example usage
file_path = 'final/data/04-01-0-Raam-Basic CNC SynDat3_Sheet_2.dxf'
parse_dxf_by_layers(file_path)
parse_dxf_blocks_and_inserts(file_path)
parse_dxf_by_entity_types(file_path)
"""