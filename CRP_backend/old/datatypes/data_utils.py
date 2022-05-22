import concurrent.futures

import torch
import pickle
from pathlib import Path
import os
import numpy as np
import cv2


import matplotlib.cm as cm
import matplotlib.colors as colors


def convert_image_type(img, target_type_min, target_type_max, target_type):
    """
    source: https://stackoverflow.com/questions/46689428/convert-np-array-of-type-float64-to-type-uint8-scaling-values
    :param img:
    :param target_type_min:
    :param target_type_max:
    :param target_type:
    :return:
    """

    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin + 1e-10)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def rescale_image(img: np.ndarray, height: int):
    """
    rescaling while preserving ratio and maximal height
    uses nearest-neighbor interpolation so that image pixel values not artifically distorted
    """

    if height == -1:
        return img

    if img.shape[0] >= img.shape[1]:
        # if height >= width
        width = int(height / img.shape[0] * img.shape[1])
    else:
        # width > height
        width = height
        height = int(width / img.shape[1] * img.shape[0])

    # in cv2 first dimension is width not height
    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_NEAREST)

def pad_image(np_image):

    """
    pad image such that resulting shape is a square, padded region is white and alpha set to transparent
    """

    height, width = np_image.shape[0:2]
    dist = abs(height - width)
    if dist == 0:
        return np_image

    # distribute padding evenly between right/left, top/bottom
    # extra pixel if not perfectly divisible

    rgba = np.concatenate((np_image, np.ones((*np_image.shape[0:2], 1))), axis=2)

    vert_extra, hor_extra = 0, 0
    if width < height:
        hor = int(dist / 2) 
        if dist % 2 != 0:
            hor_extra = 1
        vert = 0
    else:
        hor = 0
        vert = int(dist / 2) 
        if dist % 2 != 0:
            vert_extra = 1

    new_im = cv2.copyMakeBorder(rgba, vert, vert + vert_extra, hor, hor + hor_extra, cv2.BORDER_CONSTANT,
    value=(1,1,1,0))

    return new_im


def draw_heatmap(image):
    if image.shape[-1] == 1:
        # if grayscale image
        image = np.stack((image.squeeze(-1),) * 3, axis=-1)

    # paint high relevance red and low relevance blue and neutral values white
    #cmap = cm.get_cmap("RdYlGn")
    cmap = cm.get_cmap("bwr")
    norm = colors.Normalize(vmin=-1, vmax=1)

    mapping = cm.ScalarMappable(norm=norm, cmap=cmap)

    return mapping.to_rgba(image)[..., :3]


def saveFile(path, name, data):
    """
    Parameter:
        path : path to folder where to save as Path object
        name: name of file as string
        data: data to save

    Saves data in path folder. If folder does not exist, the folder is created.
    Uses pickle (Protocol 5) to save the data.
    """

    if not path.exists():

        original_umask = os.umask(0)
        try:
            os.makedirs(path, 0o0777)  # get privileges on cluster
        finally:
            os.umask(original_umask)

    if not isinstance(name, Path):
        name = Path(name)

    if type(data) is np.ndarray:
        if data.dtype == np.float64 or data.dtype == np.float32:
            data = data.astype(np.float16)

    with open(path / name, 'wb') as fp:

        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def loadFile(path, name):
    """
    Parameter:
        path : path to folder where to load as Path object
        name: name of file as string

    Loads data in path folder.
    Uses pickle (Protocol 5) to load the data.
    """

    if not path.exists():
        raise FileNotFoundError("f{name} in {path} does not exist!")

    if not isinstance(name, Path):
        name = Path(name)

    with open(path / name, 'rb') as fp:
        data = pickle.load(fp)
        return data
