import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_entities(views, info_boxes):
    fig, ax = plt.subplots(dpi=600)
    colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for _ in range(len(views))]
    plotted_labels = set()
    handles = []

    def map_linetype(linetype):
        line_type_mapping = {
            'CONTINUOUS': '-',  # Solid line
            'DASHED': (0, (5, 5)),  # Dashed line
            'DOTTED': (0, (1, 3)),  # Dotted line
            'DASHDOT': (0, (3, 5, 1, 5)),  # Dash-dot line
            'CENTER': (0, (1, 5)),  # Centerline
            'PHANTOM': (0, (5, 10, 1, 10)),  # Phantom line
            'HIDDEN': (0, (2, 5))  # Hidden line
        }
        return line_type_mapping.get(linetype.upper(), '-')  # Default to solid line if not found

    def hex_to_mpl_color(hex_color):
        if hex_color.startswith('0x'):
            hex_color = hex_color[2:]
        if len(hex_color) == 6:
            hex_color = f'#{hex_color}'
        return hex_color

    for idx, view in enumerate(views):
        label = f'View {idx + 1}' if f'View {idx + 1}' not in plotted_labels else None
        color = colors[idx]  # Get a unique color for this view
        handles.append(mpatches.Patch(color=color, label=label))
        contours = view["contours"]
        for line in contours["lines"]:
            start = line["start"]
            end = line["end"]
            ax.plot([start["x"], end["x"]], [start["y"], end["y"]], color=hex_to_mpl_color(line["colour"]),
                    linewidth=line["weight"], linestyle=map_linetype(line["style"]))

        for circle in contours["circles"]:
            center = circle["centre"]
            radius = circle["radius"]
            circle_patch = mpatches.Circle((center["x"], center["y"]), radius,
                                           edgecolor=hex_to_mpl_color(circle["colour"]), linewidth=circle["weight"],
                                           linestyle=map_linetype(circle["style"]), fill=False)
            ax.add_patch(circle_patch)

        for arc in contours["arcs"]:
            center = arc["centre"]
            radius = arc["radius"]
            start_angle = np.deg2rad(arc["start_angle"])
            end_angle = np.deg2rad(arc["end_angle"])
            arc_patch = mpatches.Arc((center["x"], center["y"]), 2*radius, 2*radius, theta1=np.rad2deg(start_angle),
                                     theta2=np.rad2deg(end_angle), edgecolor=hex_to_mpl_color(arc["colour"]),
                                     linewidth=arc["weight"], linestyle=map_linetype(arc["style"]), fill=False)
            ax.add_patch(arc_patch)

        for lwpolyline in contours["lwpolylines"]:
            points = np.array([[p["x"], p["y"]] for p in lwpolyline["points"]])
            ax.plot(points[:, 0], points[:, 1], color=hex_to_mpl_color(lwpolyline["colour"]),
                    linewidth=lwpolyline["weight"], linestyle=map_linetype(lwpolyline["style"]))

        for polyline in contours["polylines"]:
            points = np.array([[p["x"], p["y"]] for p in polyline["points"]])
            ax.plot(points[:, 0], points[:, 1], color=hex_to_mpl_color(polyline["colour"]),
                    linewidth=polyline["weight"], linestyle=map_linetype(polyline["style"]))

        for solid in contours["solids"]:
            points = np.array([[p["x"], p["y"]] for p in solid["points"]])
            polygon_patch = plt.Polygon(points, edgecolor=hex_to_mpl_color(solid["colour"]), linewidth=solid["weight"],
                                        linestyle=map_linetype(solid["style"]), fill=False)
            ax.add_patch(polygon_patch)

        for ellipse in contours["ellipses"]:
            center = ellipse["centre"]
            major_axis_length = ellipse["major_axis_length"]
            minor_axis_length = ellipse["minor_axis_length"]
            rotation_angle = ellipse["rotation_angle"]
            ellipse_patch = mpatches.Ellipse((center["x"], center["y"]), major_axis_length, minor_axis_length,
                                             angle=rotation_angle, edgecolor=hex_to_mpl_color(ellipse["colour"]),
                                             linewidth=ellipse["weight"], linestyle=map_linetype(ellipse["style"]),
                                             fill=False)
            ax.add_patch(ellipse_patch)

        for spline in contours["splines"]:
            points = np.array([[p["x"], p["y"]] for p in spline["points"]])
            ax.plot(points[:, 0], points[:, 1], color=hex_to_mpl_color(spline["colour"]), linewidth=spline["weight"],
                    linestyle=map_linetype(spline["style"]))

    ax.set_aspect('equal', 'box')
    ax.legend(loc='best', handles=handles)
    # plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0)
    # plt.figure(figsize=(10, 10))
    plt.show()


def indicate_mistakes(views):
    for view in views:
        for mistaken_line in view["mistakes"]["certain"]["lines"]:
            view["contours"]["lines"][mistaken_line]["colour"] = "0xFF0000"  # Red color
            # view["contours"]["lines"][mistaken_line]["weight"] = 2
        for potential_mistake in view["mistakes"]["potential"]["lines"]:
            view["contours"]["lines"][potential_mistake]["colour"] = "0xFFA500"  # Orange color
    return views
