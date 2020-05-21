from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg 
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation import DatasetEvaluators

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

class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
       
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
       

        return DatasetEvaluators(evaluator_list)


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





def train():

    # dataset_dicts = get_carplate_dicts()

    DatasetCatalog.register("carplate_train", lambda x='train':  get_carplate_dicts(x))
    DatasetCatalog.register("carplate_val", lambda x='val':  get_carplate_dicts(x))
    MetadataCatalog.get("carplate_val").set(thing_classes=["carplate"])
    # carplate_metadata = MetadataCatalog.get("carplate_train")

    MetadataCatalog.get("carplate_val").set(evaluator_type='coco')


    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(ROOT, CONFIG, "mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("carplate_train",)
    cfg.DATASETS.TEST = ("carplate_val")
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.DEVICE = DEVICE
    cfg.TEST.EVAL_PERIOD = 300
    cfg.MODEL.WEIGHTS = os.path.join(ROOT,WEIGHTS,"R-50.pkl")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 30000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)



    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg) 
    trainer.build_evaluator(cfg,'carplate_val',output_folder="./output/")
    trainer.resume_or_load(resume=False)
    trainer.train()

train()