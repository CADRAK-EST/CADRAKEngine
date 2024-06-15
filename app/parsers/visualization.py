import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def plot_entities(views, info_boxes):
    fig, ax = plt.subplots()
    colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for _ in range(len(views))]
    plotted_labels = set()
    handles = []

    for idx, view in enumerate(views):
        label = f'View {idx + 1}' if f'View {idx + 1}' not in plotted_labels else None
        color = colors[idx]  # Get a unique color for this view
        handles.append(mpatches.Patch(color=color, label=label))
        contours = view["contours"]

        for line in contours["lines"]:
            start = line["start"]
            end = line["end"]
            ax.plot([start["x"], end["x"]], [start["y"], end["y"]], color=color)

        for circle in contours["circles"]:
            center = circle["centre"]
            radius = circle["radius"]
            circle_patch = mpatches.Circle((center["x"], center["y"]), radius, color=color, fill=False)
            ax.add_patch(circle_patch)

        for arc in contours["arcs"]:
            center = arc["centre"]
            radius = arc["radius"]
            start_angle = np.deg2rad(arc["start_angle"])
            end_angle = np.deg2rad(arc["end_angle"])
            arc_patch = mpatches.Arc((center["x"], center["y"]), 2 * radius, 2 * radius, theta1=np.rad2deg(start_angle), theta2=np.rad2deg(end_angle), color=color)
            ax.add_patch(arc_patch)

        for lwpolyline in contours["lwpolylines"]:
            points = np.array([[p["x"], p["y"]] for p in lwpolyline["points"]])
            ax.plot(points[:, 0], points[:, 1], color=color)

        for polyline in contours["polylines"]:
            points = np.array([[p["x"], p["y"]] for p in polyline["points"]])
            ax.plot(points[:, 0], points[:, 1], color=color)

        for solid in contours["solids"]:
            points = np.array([[p["x"], p["y"]] for p in solid["points"]])
            polygon_patch = plt.Polygon(points, color=color, fill=False)
            ax.add_patch(polygon_patch)

    ax.set_aspect('equal', 'box')
    ax.legend(loc='best', handles=handles)
    plt.show()
