import cv2
import numpy as np

from osgeo import gdal

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6


def create_grid(im_shape, patch_shape, dx, dy, padding=True):
    w, h = im_shape
    ph, pw = patch_shape

    if w >= pw:
        i = np.arange(w)[:-pw + 1:dx]
        if i[-1] != w - pw:
            i = np.hstack([i, w - pw])
    else:
        i = np.array([0], dtype=np.int64)
        if not padding:
            pw = w

    if h >= ph:
        j = np.arange(h)[:-ph + 1:dy]
        if j[-1] != h - ph:
            j = np.hstack([j, h - ph])
    else:
        j = np.array([0], dtype=np.int64)
        if not padding:
            ph = h

    x, y = np.meshgrid(i, j)

    xmin = x.reshape(-1, )
    ymin = y.reshape(-1, )
    xmax = xmin + pw
    ymax = ymin + ph

    return xmin.astype(np.int), ymin.astype(np.int), xmax.astype(np.int), ymax.astype(np.int)


def init_feature(name):
    if name == 'sift':
        detector = cv2.SIFT_create(nfeatures=200)
        norm = cv2.NORM_L2
    elif name == 'surf':
        detector = cv2.xfeatures2d.SURF_create(800)
        norm = cv2.NORM_L2
    elif name == 'orb':
        detector = cv2.ORB_create(1000)
        norm = cv2.NORM_HAMMING
    elif name == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif name == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if True:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:
            flann_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6, # 12
                                key_size=12,     # 20
                                multi_probe_level=1) # 2
        matcher = cv2.FlannBasedMatcher(flann_params, {})
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio=0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)
