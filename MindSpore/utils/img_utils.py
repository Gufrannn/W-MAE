#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#The code was authored by the following people:
#
#Jaideep Pathak - NVIDIA Corporation
#Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
#Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
#Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory 
#Ashesh Chattopadhyay - Rice University 
#Morteza Mardani - NVIDIA Corporation 
#Thorsten Kurth - NVIDIA Corporation 
#David Hall - NVIDIA Corporation 
#Zongyi Li - California Institute of Technology, NVIDIA Corporation 
#Kamyar Azizzadenesheli - Purdue University 
#Pedram Hassanzadeh - Rice University 
#Karthik Kashinath - NVIDIA Corporation 
#Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import logging
import glob
from types import new_class
import random
import numpy as np
import h5py
import math
import matplotlib
import matplotlib.pyplot as plt
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.third_party_adapter.numpy_api as x2ms_np
import x2ms_adapter.torch_api.datasets as x2ms_datasets
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn
from PIL import Image, ImageColor, ImageDraw, ImageFont

class PeriodicPad2d(nn.Cell):
    """ 
        pad longitudinal (left-right) circular 
        and pad latitude (top-bottom) with zeros
    """
    def __init__(self, pad_width):
        super(PeriodicPad2d, self).__init__()
        self.pad_width = pad_width

    def construct(self, x):
        # pad left and right circular
        out = x2ms_adapter.nn_functional.x2ms_pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular") 
        # pad top and bottom zeros
        out = x2ms_adapter.nn_functional.x2ms_pad(out, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0) 
        return out

def reshape_fields(img, inp_or_tar, crop_size_x, crop_size_y,rnd_x, rnd_y, params, y_roll, train, normalize=True, orog=None, add_noise=False):
    #Takes in np array of size (n_history+1, c, h, w) and returns torch tensor of size ((n_channels*(n_history+1), crop_size_x, crop_size_y)

    if len(np.shape(img)) ==3:
        img = np.expand_dims(img, 0)

    
    img = img[:, :, 0:720] #remove last pixel
    n_history = np.shape(img)[0] - 1
    img_shape_x = np.shape(img)[-2]
    img_shape_y = np.shape(img)[-1]
    n_channels = np.shape(img)[1] #this will either be N_in_channels or N_out_channels
    channels = params.in_channels if inp_or_tar =='inp' else params.out_channels
    means = np.load(params.global_means_path)[:, channels]
    stds = np.load(params.global_stds_path)[:, channels]
    if crop_size_x == None:
        crop_size_x = img_shape_x
    if crop_size_y == None:
        crop_size_y = img_shape_y

    if normalize:
        if params.normalization == 'minmax':
            raise Exception("minmax not supported. Use zscore")
        elif params.normalization == 'zscore':
            img -=means
            img /=stds

    if params.add_grid:
        if inp_or_tar == 'inp':
            if params.gridtype == 'linear':
                assert params.N_grid_channels == 2, "N_grid_channels must be set to 2 for gridtype linear"
                x = np.meshgrid(np.linspace(-1, 1, img_shape_x))
                y = np.meshgrid(np.linspace(-1, 1, img_shape_y))
                grid_x, grid_y = np.meshgrid(y, x)
                grid = np.stack((grid_x, grid_y), axis = 0)
            elif params.gridtype == 'sinusoidal':
                assert params.N_grid_channels == 4, "N_grid_channels must be set to 4 for gridtype sinusoidal"
                x1 = np.meshgrid(x2ms_adapter.tensor_api.sin(np, np.linspace(0, 2*np.pi, img_shape_x)))
                x2 = np.meshgrid(x2ms_adapter.tensor_api.cos(np, np.linspace(0, 2*np.pi, img_shape_x)))
                y1 = np.meshgrid(x2ms_adapter.tensor_api.sin(np, np.linspace(0, 2*np.pi, img_shape_y)))
                y2 = np.meshgrid(x2ms_adapter.tensor_api.cos(np, np.linspace(0, 2*np.pi, img_shape_y)))
                grid_x1, grid_y1 = np.meshgrid(y1, x1)
                grid_x2, grid_y2 = np.meshgrid(y2, x2)
                grid = np.expand_dims(np.stack((grid_x1, grid_y1, grid_x2, grid_y2), axis = 0), axis = 0)
            img = x2ms_np.concatenate((img, grid), axis = 1 )

    if params.orography and inp_or_tar == 'inp':
        img = x2ms_np.concatenate((img, np.expand_dims(orog, axis = (0,1) )), axis = 1)
        n_channels += 1

    if params.roll:
        img = np.roll(img, y_roll, axis = -1)

    if train and (crop_size_x or crop_size_y):
        img = img[:,:,rnd_x:rnd_x+crop_size_x, rnd_y:rnd_y+crop_size_y]

    if inp_or_tar == 'inp':
        img = np.reshape(img, (n_channels*(n_history+1), crop_size_x, crop_size_y))
    elif inp_or_tar == 'tar':
        if params.two_step_training:
            img = np.reshape(img, (n_channels*2, crop_size_x, crop_size_y))
        else:
            img = np.reshape(img, (n_channels, crop_size_x, crop_size_y))

    if add_noise:
        img = img + np.random.normal(0, scale=params.noise_std, size=img.shape)

    # return x2ms_adapter.as_tensor(img)
    return img
          
def reshape_precip(img, inp_or_tar, crop_size_x, crop_size_y,rnd_x, rnd_y, params, y_roll, train, normalize=True):

    if len(np.shape(img)) ==2:
        img = np.expand_dims(img, 0)

    img = img[:,:720,:]
    img_shape_x = img.shape[-2]
    img_shape_y = img.shape[-1]
    n_channels = 1
    if crop_size_x == None:
        crop_size_x = img_shape_x
    if crop_size_y == None:
        crop_size_y = img_shape_y

    if normalize:
        eps = params.precip_eps
        img = np.log1p(img/eps)
    if params.add_grid:
        if inp_or_tar == 'inp':
            if params.gridtype == 'linear':
                assert params.N_grid_channels == 2, "N_grid_channels must be set to 2 for gridtype linear"
                x = np.meshgrid(np.linspace(-1, 1, img_shape_x))
                y = np.meshgrid(np.linspace(-1, 1, img_shape_y))
                grid_x, grid_y = np.meshgrid(y, x)
                grid = np.stack((grid_x, grid_y), axis = 0)
            elif params.gridtype == 'sinusoidal':
                assert params.N_grid_channels == 4, "N_grid_channels must be set to 4 for gridtype sinusoidal"
                x1 = np.meshgrid(x2ms_adapter.tensor_api.sin(np, np.linspace(0, 2*np.pi, img_shape_x)))
                x2 = np.meshgrid(x2ms_adapter.tensor_api.cos(np, np.linspace(0, 2*np.pi, img_shape_x)))
                y1 = np.meshgrid(x2ms_adapter.tensor_api.sin(np, np.linspace(0, 2*np.pi, img_shape_y)))
                y2 = np.meshgrid(x2ms_adapter.tensor_api.cos(np, np.linspace(0, 2*np.pi, img_shape_y)))
                grid_x1, grid_y1 = np.meshgrid(y1, x1)
                grid_x2, grid_y2 = np.meshgrid(y2, x2)
                grid = np.expand_dims(np.stack((grid_x1, grid_y1, grid_x2, grid_y2), axis = 0), axis = 0)
            img = x2ms_np.concatenate((img, grid), axis = 1 )

    if params.roll:
        img = np.roll(img, y_roll, axis = -1)

    if train and (crop_size_x or crop_size_y):
        img = img[:,rnd_x:rnd_x+crop_size_x, rnd_y:rnd_y+crop_size_y]

    img = np.reshape(img, (n_channels, crop_size_x, crop_size_y))
    # return x2ms_adapter.as_tensor(img)
    return img


def vis_precip(fields):
    pred, tar = fields
    fig, ax = plt.subplots(1, 2, figsize=(24,12))
    ax[0].imshow(pred, cmap="coolwarm")
    ax[0].set_title("tp pred")
    ax[1].imshow(tar, cmap="coolwarm")
    ax[1].set_title("tp tar")
    fig.tight_layout()
    return fig

def save_image(img, img_path):
    """Save a numpy image to the disk

    Parameters:
        img (numpy array / Tensor): image to save.
        image_path (str): the path of the image.
    """
    if isinstance(img, mindspore.Tensor):
        img = img.asnumpy()
    elif not isinstance(img, np.ndarray):
        raise ValueError("img should be Tensor or numpy array, but get {}".format(type(img)))
    img = decode_image(img)

    img_pil = Image.fromarray(img)
    img_pil.save(img_path)

def decode_image(img):
    if img.ndim==2:
        img = np.expand_dims(img, axis=0)

    if img.shape[0] == 1:  # if single-channel, convert to 3-channel
        img = np.concatenate((img, img, img), 0)

    return np.clip(img*255+0.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)