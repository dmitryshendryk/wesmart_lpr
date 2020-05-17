import torch 
torch.cuda.set_device(0)

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

ROOT = os.path.abspath('./')
DATA_FOLDER = 'data/plates_with_json'
CONFIG = 'config'
WEIGHTS = 'weights'
DEVICE = 'cuda'

def get_carplate_dicts():
    path = os.path.join(ROOT, DATA_FOLDER)
    json_file = os.path.join(path, "dataset_mask_full_no_aug.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)
    
    dataset_dicts = []
    for idx, v in enumerate(list(imgs_anns['_via_img_metadata'].values())):
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

dataset_dicts = get_carplate_dicts()


DatasetCatalog.register("carplate", lambda : get_carplate_dicts())
MetadataCatalog.get("carplate").set(thing_classes=["carplate"])
carplate_metadata = MetadataCatalog.get("carplate_train")

cfg = get_cfg()
cfg.merge_from_file(os.path.join(ROOT, CONFIG, "mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("carplate",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.DEVICE = DEVICE
cfg.MODEL.WEIGHTS = os.path.join(ROOT,WEIGHTS,"model_final.pth")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
# trainer.train()


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("carplate", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "carplate")
inference_on_dataset(trainer.model, val_loader, evaluator)
# another equivalent way is to use trainer.test
