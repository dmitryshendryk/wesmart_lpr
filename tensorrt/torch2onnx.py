

import os

import numpy as np
import cv2

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu


import torch
import numpy as np

import torch.onnx
import onnx 
import onnxruntime

from PIL import Image
import torchvision.transforms as transforms


torch_model = torch.load('../models/segmentation_unet/best_model.pth', map_location=torch.device('cuda'))

x = torch.randn(1, 3, 1024, 576, requires_grad=True)
torch_out = torch_model(x)

torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "../models/segmentation_unet/super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
