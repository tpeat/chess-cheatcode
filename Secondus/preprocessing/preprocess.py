import numpy as np
import re
from skimage.util.shape import view_as_blocks
from skimage import transform as sktransform
from skimage import io
import os


def process_image(img, downsample_size = 200):
    square_size = int(downsample_size/8)
    img = sktransform.resize(io.imread(img), 
                                  (downsample_size, downsample_size), 
                                  mode='constant')
    tiles = view_as_blocks(img, block_shape=(square_size, 
                                                  square_size, 
                                                  3)).squeeze(axis=2)
    return tiles.reshape(64, square_size, square_size, 3), img  