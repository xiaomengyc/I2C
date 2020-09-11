
import torch
from .._ext import sp_segment_func, sp_atten_func
import pdb

def get_sp_mask(heatmaps, superpixel, th_high=0.4, th_low=0.1):

    segments = superpixel.new_full(superpixel.size(), 0)

    biggest_seg = superpixel.max().item()
    sp_segment_func(heatmaps.float(), superpixel, segments, biggest_seg, th_high, th_low)

    return segments

def get_sp_atten(heatmaps, superpixel):

    segments = superpixel.new_full(superpixel.size(), 0)

    biggest_seg = superpixel.max().item()

    sp_atten_func(heatmaps.float(), superpixel, segments, biggest_seg)

    return segments
