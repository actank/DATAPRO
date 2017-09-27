import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff

import matplotlib.pyplot as plt
from matplotlib import cm

FILE_2015 = '../data/quickbird2015.tif'
FILE_2017 = '../data/quickbird2017.tif'
FILE_cadastral2015 = '../data/cadastral2015.tif'
FILE_tinysample = '../data/tinysample.tif'

#im_2015 = tiff.imread(FILE_2015).transpose([1, 2, 0])

#im_2017 = tiff.imread(FILE_2017).transpose([1, 2, 0])

im_tiny = tiff.imread(FILE_tinysample)

im_cada = tiff.imread(FILE_cadastral2015)

#im_2015.shape

im_tiny.shape

#im_cada.shape


def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 6))

p1 = plt.subplot(121)
i1 = p1.imshow(scale_percentile(im_tiny[100:1000, 100:1000, :3]))
plt.colorbar(i1)

p2 = plt.subplot(122)
i2 = p2.imshow(im_tiny[100:1000, 100:1000, 2])
plt.colorbar(i2)

plt.show()
