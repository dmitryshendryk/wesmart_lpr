# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
"""
VoVNet Training Script.
This script is a simplified version of the training script in detectron2/tools.
"""

import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    COCOEvaluator, 
    COCOPanopticEvaluator,
    SemSegEvaluator,
    DatasetEvaluators,
    verify_results
    )
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode

import sys  
sys.path.insert(0, '/Users/dmitry/Documents/Business/Projects/Upwork/Wesmart/wesmart_lpr')

from models.center_mask.vovnet_config import add_vovnet_config
import cv2
import numpy as np
import json
import os


ROOT = os.path.abspath('../../')
CONFIG = 'config'
DATA_FOLDER = 'data/plates_with_json'
WEIGHTS = 'weights'


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        
        return DatasetEvaluators(evaluator_list)


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




def setup(args):
    """
    Create configs and perform basic setups.
    """

    dataset_dicts = get_carplate_dicts()

    DatasetCatalog.register("carplate", lambda : get_carplate_dicts())
    MetadataCatalog.get("carplate").set(thing_classes=["carplate"])
    carplate_metadata = MetadataCatalog.get("carplate")


    cfg = get_cfg()
    add_vovnet_config(cfg)
    cfg.merge_from_file(os.path.join(ROOT, CONFIG, "faster_rcnn_V_39_FPN_3x.yaml"))
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TRAIN = ("carplate",)
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = os.path.join(ROOT,WEIGHTS,"vovnet39_ese_detectron2.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    # cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    # if args.eval_only:
    #     model = Trainer.build_model(cfg)
    #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=args.resume
    #     )
    #     res = Trainer.test(cfg, model)
    #     if comm.is_main_process():
    #         verify_results(cfg, res)
    #     return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )