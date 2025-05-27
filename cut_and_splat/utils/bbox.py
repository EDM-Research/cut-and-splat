import numpy as np


def generate_bounding_box(bbox, image, width, height):
    x1, y1, x2, y2 = bbox

    xc = x1 + (x2 - x1) / 2
    yc = y1 + (y2 - y1) / 2

    x1 = max(xc - width / 2, 0)
    y1 = max(yc - height / 2, 0)

    x2 = min(x1 + width, image.shape[1])
    y2 = min(y1 + height, image.shape[0])

    return int(x1), int(y1), int(x2), int(y2)


def get_enclosing_bounding_box(image: np.array):
    if len(image.shape) > 2:
        image = np.sum(image, axis=-1)
    where = np.array(np.where(image))
    if where.size == 0:
        return 0, 0, 0, 0
    y1, x1 = np.amin(where, axis=1)
    y2, x2 = np.amax(where, axis=1)
    return int(x1), int(y1), int(x2), int(y2)


def get_bbox_center(bbox: list):
    x1, y1, x2, y2 = bbox

    return x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2