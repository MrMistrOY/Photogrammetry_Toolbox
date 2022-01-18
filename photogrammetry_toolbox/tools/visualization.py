import matplotlib.pyplot as plt

from matplotlib.patches import Polygon


def draw_obj_image(img, objects, color):
    plt.figure()
    plt.imshow(img)
    ax = plt.gca()
    id_group = None
    group_xy = []
    for obj in objects:
        type_obj = obj['type_obj']
        pixel_xy = obj['pixel']
        id_obj = obj['id_obj']
        if type_obj == 'point':
            ax.scatter(pixel_xy[0], pixel_xy[1], s=40, c=color, marker='o')
        elif type_obj == 'region':
            if id_group is None:
                id_group = id_obj
            if id_group == id_obj:
                group_xy.append(pixel_xy)
            else:
                ax.add_patch(Polygon(group_xy, fill=True, edgecolor=(0, 1, 0, 1), hatch='/',
                                     facecolor=(0, 0.4, 0, 0.3), linewidth=3))
                id_group = id_obj
                group_xy = [pixel_xy]
    plt.show()
