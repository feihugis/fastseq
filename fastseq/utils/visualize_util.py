# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for data visualization"""

import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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


def plot_hypo_beams(hypo_beams, img_path, title, dpi=300):
    num_beams = len(hypo_beams)
    color_map = plt.cm.get_cmap('Paired', len(hypo_beams))
    plt.rcParams["figure.figsize"] = (20, 20)
    for i, beam in enumerate(hypo_beams):
        score, hypo, score_tracking = beam
        assert score_tracking.dim() == 1
        score_tracking = score_tracking.numpy()
        plt.plot(
            range(score_tracking.shape[0] - 1),
            score_tracking[1:],
            c=color_map(i),
            label="{}_{}".format(i, score))
    plt.xlabel('Step')
    plt.ylabel('Log Probability')
    plt.title(title)
    plt.legend(range(num_beams))
    plt.savefig(img_path, dpi=dpi)
    plt.close()


def plot_beam_score_trackings(beam_score_trackings, rouges, img_path, dpi=100):
    test_num_beams = list(beam_score_trackings.keys())
    num_samples = len(beam_score_trackings[test_num_beams[0]])
    color_map = plt.cm.get_cmap('Paired', len(test_num_beams))
    plt.rcParams["figure.figsize"] = (20, 20*num_samples)
    for sample_id in range(num_samples):
        plt.subplot(num_samples, 1, sample_id + 1)
        for i, num_beams in enumerate(test_num_beams):
            beam_score_tracking = beam_score_trackings[num_beams][sample_id].cpu().numpy()
            plt.plot(range(
                beam_score_tracking.shape[0] - 1),
                beam_score_tracking[1:],
                c=color_map(i),
                label=num_beams)
            rouge = rouges[num_beams][sample_id]
            plt.text(
                18, -i,
                "nb={}: score={:.4f}, lp_score={:.4f}, (r1: {:.4f}, r2: {:.4f}, rL: {:.4f})".format(
                    num_beams,
                    beam_score_tracking[-2],
                    beam_score_tracking[-1],
                    rouge['rouge1'], rouge['rouge2'], rouge['rougeL']),
                wrap=True)
        plt.xlabel('Step')
        plt.ylabel('Log Probability')
        plt.title("{}th sample".format(sample_id))
        plt.legend(test_num_beams)

    plt.savefig(img_path, dpi=dpi)
    plt.close()

