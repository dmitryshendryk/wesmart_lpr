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

def get_carplate_dicts(mode):
    path = os.path.join(ROOT, DATA_FOLDER)
    json_file = os.path.join(path, "dataset_mask_full_no_aug.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)
    
    dataset_dicts = []
    dataset_len = len(list(imgs_anns['_via_img_metadata'].values()))
    dataset = list(imgs_anns['_via_img_metadata'].values())
    if mode == 'train':
        dataset = dataset[:dataset_len - int(dataset_len*0.1)]
    elif mode == 'val':
        dataset = dataset[dataset_len - int(dataset_len*0.1):]

    print(len(dataset))
    for idx, v in enumerate(list(dataset)):
        record = {}
        
        filename = os.path.join(path, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for anno in annos:
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


DatasetCatalog.register("carplate_train", lambda x='train':  get_carplate_dicts(x))
DatasetCatalog.register("carplate_val", lambda x='val':  get_carplate_dicts(x))
MetadataCatalog.get("carplate_val").set(thing_classes=["carplate"])
carplate_metadata = MetadataCatalog.get("carplate_val")

MetadataCatalog.get("carplate_val").set(evaluator_type='coco')


cfg = get_cfg()
cfg.merge_from_file(os.path.join(ROOT, CONFIG, "mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("carplate_train",)
cfg.DATASETS.TEST = ("carplate_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.DEVICE = DEVICE
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "maskrcnn_model_final_20200522.pth")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
# trainer.resume_or_load(resume=False)
# trainer.train()

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("carplate_val", )
predictor = DefaultPredictor(cfg)

import time
times = []
dataset_dicts = get_carplate_dicts('val')
im = cv2.imread(dataset_dicts[0]["file_name"])
for i in range(20):
    start_time = time.time()
    outputs = predictor(im)
    delta = time.time() - start_time
    times.append(delta)
mean_delta = np.array(times).mean()
fps = 1 / mean_delta
print("Average(sec):{:.2f},fps:{:.2f}".format(mean_delta, fps))