import networkx as nx
from app.parsers.utilities import format_point, is_close
import logging

logger = logging.getLogger(__name__)


def process_dimensions_to_graphs(dimensions, scale=1e3):
    g_x = nx.Graph()
    g_y = nx.Graph()
    g_xy = nx.Graph()
    circles_info = {}
    dimension_list = []

    for dim in dimensions:
        if dim.dxf.hasattr('defpoint4'):
            start = format_point((dim.dxf.get('defpoint', None).x, dim.dxf.get('defpoint', None).y), scale)
            end = format_point((dim.dxf.get('defpoint4', None).x, dim.dxf.get('defpoint4', None).y), scale)
            centre = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)  # Center as midpoint of start and end points

            dimension_dict = {
                'length': round(dim.dxf.get('actual_measurement', None)),
                'line_type': 'circular',
                'centre': centre
            }
            circles_info[dimension_dict["length"]/2] = dimension_dict["centre"]

        else:
            startpoint = dim.dxf.get('defpoint2', None)
            endpoint = dim.dxf.get('defpoint3', None)
            dimension_dict = {
                'start': format_point((startpoint.x, startpoint.y), scale),
                'end': format_point((endpoint.x, endpoint.y), scale),
                'length': round(dim.dxf.get('actual_measurement', None), 3), 'line_type': 'linear',
                "middle": format_point(((startpoint.x + endpoint.x) / 2, (startpoint.y + endpoint.y) / 2), scale)
            }
            if dimension_dict["start"][0] == dimension_dict["end"][0]:
                g_y.add_edge(dimension_dict["start"][1], dimension_dict["middle"][1], weight=dimension_dict["length"]/2)
                g_y.add_edge(dimension_dict["start"][1], dimension_dict["middle"][1], weight=dimension_dict["length"]/2)
                g_y.add_edge(dimension_dict["middle"][1], dimension_dict["end"][1], weight=dimension_dict["length"]/2)
                dimension_dict["geometry"] = "vertical"
            elif dimension_dict["start"][1] == dimension_dict["end"][1]:
                g_x.add_edge(dimension_dict["start"][0], dimension_dict["middle"][0], weight=dimension_dict["length"]/2)
                g_x.add_edge(dimension_dict["middle"][0], dimension_dict["end"][0], weight=dimension_dict["length"]/2)
                dimension_dict["geometry"] = "horizontal"
            else:
                if dimension_dict["middle"][0] == dim.dxf.get('text_midpoint', None).x or \
                        dimension_dict["middle"][1] == dim.dxf.get('text_midpoint', None).y:
                    g_xy.add_edge(dimension_dict["start"], dimension_dict["middle"], weight=dimension_dict["length"]/2)
                    g_xy.add_edge(dimension_dict["middle"], dimension_dict["end"], weight=dimension_dict["length"]/2)
                    dimension_dict["geometry"] = "diagonal"
                else:
                    checkpoint = dim.dxf.get('defpoint', None)
                    checkpoint = format_point((checkpoint.x, checkpoint.y), scale)
                    # print("Checkpoint:", checkpoint)
                    if checkpoint[0] == dimension_dict["start"][0] or checkpoint[0] == dimension_dict["end"][0]:
                        dimension_dict["geometry"] = "horizontal"
                        g_x.add_edge(dimension_dict["start"][0], dimension_dict["middle"][0], weight=dimension_dict["length"]/2)
                        g_x.add_edge(dimension_dict["middle"][0], dimension_dict["end"][0], weight=dimension_dict["length"]/2)
                    elif checkpoint[1] == dimension_dict["start"][1] or checkpoint[1] == dimension_dict["end"][1]:
                        dimension_dict["geometry"] = "vertical"
                        g_y.add_edge(dimension_dict["start"][1], dimension_dict["middle"][1], weight=dimension_dict["length"]/2)
                        g_y.add_edge(dimension_dict["middle"][1], dimension_dict["end"][1], weight=dimension_dict["length"]/2)
                    else:
                        #logger.error("Dimension %s is not vertical, horizontal or diagonal", dimension_dict)
                        dimension_dict["geometry"] = "unknown"

        dimension_list.append(dimension_dict)
        # print(dimension_list[-1])

    return {"g_x": g_x, "g_y": g_y, "g_xy": g_xy, "circles_info": circles_info}


def find_lengths(dimensions, lines, circles):
    view_dimensions_graphs = process_dimensions_to_graphs(dimensions)
    ids_of_mistaken_lines = []
    ids_of_potential_mistaken_lines = []
    ids_of_mistaken_circles = []
    for i, line in enumerate(lines):
        if "visible" not in line["layer"].lower():  # Check only actual contours
            continue
        start = line["start"]
        end = line["end"]
        try:
            if start["x"] == end["x"]:
                shortest_path = nx.shortest_path(view_dimensions_graphs["g_y"], source=start["y"], target=end["y"],
                                                 weight='weight')
                logger.info("Vertical path found: %s", shortest_path)
            elif start["y"] == end["y"]:
                shortest_path = nx.shortest_path(view_dimensions_graphs["g_x"], source=start["x"], target=end["x"],
                                                 weight='weight')
                logger.info("Horizontal path found: %s", shortest_path)
            else:
                shortest_path = nx.shortest_path(view_dimensions_graphs["g_xy"], source=start, target=end,
                                                 weight='weight')
                logger.info("Diagonal path found: %s", shortest_path)
        except nx.NetworkXNoPath:
            logger.warning("Mistake found, no dimension can be calculated for line: %s, %s", start, end)
            ids_of_mistaken_lines.append(i)
        except nx.NodeNotFound:
            logger.warning("Mistake or wrong line input, no endpoints found in graph for line: %s, %s", start, end)
            ids_of_potential_mistaken_lines.append(i)
    for i, circle in enumerate(circles):
        if "visible" not in circle["layer"].lower():  # Check only actual contours
            continue
        # Simple check to see whether a CIRCLE entity's radius is in the circles_info dictionary
        if circle["radius"] in view_dimensions_graphs["circles_info"]:
            logger.info("Circle with real centre %s and radius %s, corresponding to centre %s",
                        circle["centre"], circle["radius"], view_dimensions_graphs["circles_info"][circle["radius"]])
        else:
            logger.warning("Circle with real centre %s and radius %s not found in circles_info", circle["centre"],
                           circle["radius"])
            ids_of_mistaken_circles.append(i)
    return ids_of_mistaken_lines, ids_of_potential_mistaken_lines, ids_of_mistaken_circles


"""
# Define the start and end points
start_point = (2, 3)
end_point = (3, 4)

# Find the shortest path based on known lengths
shortest_path = nx.shortest_path(G, source=start_point, target=end_point, weight='weight')

# Calculate the total length of the path
path_length = sum(G[u][v]['weight'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))

print(f"The length between point {start_point} and point {end_point} is: {path_length}")
"""
