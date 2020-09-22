# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for data visualization"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy


def _normalize(arr):
    min_v = arr.min()
    max_v = arr.max()
    normalized_arr = (arr - min_v) / (max_v - min_v)
    return normalized_arr

def save_tensor_to_img(data, img_path):
    data_np = data.cpu().numpy()
    normalized_data = _normalize(data_np) * 255
    Image.fromarray(normalized_data).convert("L").save(img_path)

def plot_tensor(data, img_path, max_lines=4):
    np_data = data.cpu().numpy()
    # np_data = _normalize(np_data)
    height, width = np_data.shape
    height = min(height, max_lines)
    color_map = plt.cm.get_cmap('Accent', height)
    labels = range(height)
    x = range(width)
    for r in reversed(range(height)):
        plt.plot(x, np_data[r, :], c=color_map(r), label=r)

    plt.legend(labels)
    plt.savefig(img_path, dpi=800)
    plt.close()

