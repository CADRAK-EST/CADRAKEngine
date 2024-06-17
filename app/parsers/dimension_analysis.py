import networkx as nx
from utilities import normalize_point, is_close, normalize_point2


def process_dimensions_to_graphs(dimensions, scale=1e3):
    G_x = nx.Graph()
    G_y = nx.Graph()
    G_xy = nx.Graph()
    circles_info = {}
    dimension_list = []

    for dim in dimensions:
        if dim.dxf.hasattr('defpoint4'):
            start = normalize_point((dim.dxf.get('defpoint', None).x, dim.dxf.get('defpoint', None).y), scale)
            end = normalize_point((dim.dxf.get('defpoint4', None).x, dim.dxf.get('defpoint4', None).y), scale)
            centre = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)  # Center as midpoint of start and end points

            dimension_dict = {
                'length': round(dim.dxf.get('actual_measurement', None)),
                'line_type': 'circular',
                'centre': centre
            }
            circles_info[dimension_dict["length"]/2] = dimension_dict["centre"]

        else:
            dimension_dict = {
                'start': normalize_point((dim.dxf.get('defpoint2', None).x, dim.dxf.get('defpoint2', None).y), scale),
                'end': normalize_point((dim.dxf.get('defpoint3', None).x, dim.dxf.get('defpoint3', None).y), scale),
                'length': round(dim.dxf.get('actual_measurement', None)),
                'line_type': 'linear'
            }
            if is_close(dimension_dict["start"][0], dimension_dict["end"][0], tol=1/scale):
                G_y.add_edge(dimension_dict["start"][1], dimension_dict["end"][1], weight=dimension_dict["length"])
                dimension_dict["geometry"] = "vertical"
            elif is_close(dimension_dict["start"][1], dimension_dict["end"][1], tol=1/scale):
                G_x.add_edge(dimension_dict["start"][0], dimension_dict["end"][0], weight=dimension_dict["length"])
                dimension_dict["geometry"] = "horizontal"
            else:
                G_xy.add_edge(dimension_dict["start"], dimension_dict["end"], weight=dimension_dict["length"])
                dimension_dict["geometry"] = "diagonal"

        dimension_list.append(dimension_dict)
        #print(dimension_list[-1])

    return {"G_x": G_x, "G_y": G_y, "G_xy": G_xy, "circles_info": circles_info}


def find_lengths(dimensions, lines, circles):
    view_dimensions_graphs = process_dimensions_to_graphs(dimensions)
    ids_of_mistaken_lines = []
    for i, line in enumerate(lines):
        start = line["start"]
        end = line["end"]
        try:
            if start["x"] == end["x"]:
                shortest_path = nx.shortest_path(view_dimensions_graphs["G_y"], source=start["y"], target=end["y"], weight='weight')
                print("Vertical path found:", shortest_path)
            elif start["y"] == end["y"]:
                shortest_path = nx.shortest_path(view_dimensions_graphs["G_x"], source=start["x"], target=end["x"], weight='weight')
                print("Horizontal path found:", shortest_path)
            else:
                shortest_path_x = nx.shortest_path(view_dimensions_graphs["G_x"], source=start["x"], target=end["x"], weight='weight')
                shortest_path_y = nx.shortest_path(view_dimensions_graphs["G_y"], source=start["y"], target=end["y"], weight='weight')
                shortest_path = shortest_path_x + shortest_path_y
                print("Diagonal path found:", shortest_path)
        except nx.NetworkXNoPath:
            print("Mistake found, no dimension can be calculated for line:", start, end)
            ids_of_mistaken_lines.append(i)
        except nx.NodeNotFound:
            print("Mistake or wrong line input, no endpoints found in graph for line:", start, end)
            ids_of_mistaken_lines.append(i)
    for circle in circles:
        # Simple check to see whether a CIRCLE entity's radius is in the circles_info dictionary
        if circle["radius"] in view_dimensions_graphs["circles_info"]:
            print("Circle with real centre", circle["centre"], "and radius", str(circle["radius"]) +
                  ", corresponding to centre",
                  view_dimensions_graphs["circles_info"][circle["radius"]])
        else:
            print("Circle with real centre", circle["centre"], "and radius", circle["radius"],
                  "not found in circles_info")
    return ids_of_mistaken_lines


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
