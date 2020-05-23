from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg 
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

import json
import os
import sys
import random 
import PIL
import matplotlib.pyplot as plt 
import cv2
import numpy as np

ROOT = os.path.abspath('../../')
DATA_FOLDER = 'data/plates_with_json'
CONFIG = 'config'
WEIGHTS = 'weights'
DEVICE = 'cuda'

sys.path.append(ROOT)

from data_handler.dataset_handler import get_carplate_dicts

DatasetCatalog.register("carplate_train", lambda x='train':  get_carplate_dicts(x, ROOT))
DatasetCatalog.register("carplate_val", lambda x='val':  get_carplate_dicts(x, ROOT))
MetadataCatalog.get("carplate_val").set(thing_classes=["carplate"])
carplate_metadata = MetadataCatalog.get("carplate_val")

MetadataCatalog.get("carplate_val").set(evaluator_type='coco')


cfg = get_cfg()
cfg.DATASETS.TRAIN = ("carplate_train",)
cfg.DATASETS.TEST = ("carplate_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.DEVICE = DEVICE
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "centermask_model_final_20200522.pth")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  
cfg.SOLVER.MAX_ITER = 300    
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  
cfg.DATASETS.TEST = ("carplate_val", )
predictor = DefaultPredictor(cfg)

import time
times = []
dataset_dicts = get_carplate_dicts('val',ROOT)
im = cv2.imread(dataset_dicts[0]["file_name"])
for i in range(20):
    start_time = time.time()
    outputs = predictor(im)
    delta = time.time() - start_time
    times.append(delta)
mean_delta = np.array(times).mean()
fps = 1 / mean_delta
print("Average(sec):{:.2f},fps:{:.2f}".format(mean_delta, fps))