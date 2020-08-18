#%%
# import some common libraries
import torch, torchvision
import numpy as np
import pandas as pd
import cv2
import os
import random
import logging
import json
from tqdm import tqdm
from matplotlib import pyplot as plt

# import some detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator

# importing custom classes from the repo
from DataLoader import DataLoader
from ImageDisplayer import ImageDisplayer
from CustomVisualiser import CustomVisualizer
from CustomTrainer import CustomTrainer

# %%
root_dir = "./../Data" # change this to download to a specific location on your pc
DataLoader().download_datasets(root_dir)
DataLoader().generateAllJsonDataAnnotations(root_dir)

# %%
# to decide which data should be loaded use this:

# type_of_annotation = ["system_measures"]
# type_of_annotation = ["stave_measures"]
type_of_annotation = ["staves"]

# type_of_annotation = ["system_measures", "staves"]
# type_of_annotation = ["system_measures", "stave_measures", "staves"]

json_pathname_extension = "-".join(str(elem) for elem in type_of_annotation)

# %%
json_path = os.path.join(root_dir, "CVC_muscima_" + json_pathname_extension + ".json")

muscima_data = DataLoader().load_from_json(json_path)
DataLoader().show_data(muscima_data, type_of_annotation)

# %%
json_path = os.path.join(root_dir, "AudioLabs_" + json_pathname_extension + ".json")

audioLabs_data = DataLoader().load_from_json(json_path)
DataLoader().show_data(audioLabs_data, type_of_annotation)

# %%
def registerDataset(data_name, d, data, classes):
    DatasetCatalog.register(data_name, lambda d=d: data)
    MetadataCatalog.get(data_name).set(thing_classes=classes)

    return MetadataCatalog.get(data_name)

# %%
from sklearn.model_selection import train_test_split

musicma_train_data, test_val_data = train_test_split(muscima_data, test_size=0.4, random_state=1)
musicma_test_data, musicma_val_data = train_test_split(test_val_data, test_size=0.5, random_state=1)

audiolabs_train_data, test_val_data = train_test_split(audioLabs_data, test_size=0.4, random_state=1)
audiolabs_test_data, audiolabs_val_data = train_test_split(test_val_data, test_size=0.5, random_state=1)

train_data = musicma_train_data + audiolabs_train_data
test_data = musicma_test_data + audiolabs_test_data
val_data = musicma_val_data + audiolabs_val_data

train_data_name = "train"
metadata = registerDataset(train_data_name, train_data_name, train_data, type_of_annotation)

test_data_name = "test"
registerDataset(test_data_name, test_data_name, test_data, type_of_annotation)

val_data_name = "val"
registerDataset(val_data_name, val_data_name, val_data, type_of_annotation)

# %%
def setup_cfg(train_data_name, test_data_name, val_period, max_iter, num_classes, model_output_dir, cfg_file, existing_model_weight_path=None):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))

    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)
    cfg.OUTPUT_DIR = model_output_dir

    setup_logger(model_output_dir)

    cfg.DATASETS.TRAIN = (train_data_name,)
    cfg.DATASETS.TEST = (test_data_name,)

    # TODO: how about unix / mac?
    if sys.platform.startswith("linux"):
        cfg.DATALOADER.NUM_WORKERS = 4 # Number of data loading threads
    else:
        # has to be 0 for windows see: https://github.com/pytorch/pytorch/issues/2341
        cfg.DATALOADER.NUM_WORKERS = 0 # Number of data loading threads

    if existing_model_weight_path:
        cfg.MODEL.WEIGHTS = existing_model_weight_path
        continue_training = True
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_file)  # Let training initialize from model zoo
        continue_training = False

    cfg.TEST.EVAL_PERIOD = val_period

    # Number of images per batch across all machines.
    # If we have 16 GPUs and IMS_PER_BATCH = 32,
    # each GPU will see IMS_PER_BATCH images per batch.
    # less images per batch -> less gpu memory
    # but will also need more iterations
    cfg.SOLVER.IMS_PER_BATCH = 3
    cfg.SOLVER.BASE_LR = 0.005  # pick a good LR
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD = val_period # Save a checkpoint after every this number of iterations
    cfg.SOLVER.GAMMA = 0.8
    cfg.SOLVER.WARMUP_ITERS = 1200

    # to decrease the learning rate at set iterations, use this code here
    # manually
    # steps = (3300, 4500, 6000, 7800, 9900, 12300, 15000, 18000)
    # or with a bit of extra code in a loop
    # steps = ()
    # for i in range(cfg.SOLVER.WARMUP_ITERS * 2, val_period * 100, val_period * 3): # decrease lr every 3 steps iteration (every 3 "epochs")
    #     steps = steps + (i, )
    # cfg.SOLVER.STEPS = steps

    cfg.SOLVER.STEPS = ()

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # 128 faster, and good enough for toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set the testing threshold for this model. Model should be at least 20% confident detection is correct
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.2

    # Set seed to negative to fully randomize everything.
    # Set seed to positive to use a fixed seed. Note that a fixed seed increases
    # reproducibility but does not guarantee fully deterministic behavior.
    # Disabling all parallelism further increases reproducibility.
    cfg.SEED = 1

    return cfg, continue_training

# %%
max_iter = 20000

# val_period has to be dividable by 20 - see: https://github.com/facebookresearch/detectron2/issues/1714
# because the PeriodicWriter hook defaults to logging every 20 iterations in https://github.com/facebookresearch/detectron2/blob/439134dd6fe7c7aa0df4571d27a6386c1678551f/detectron2/engine/hooks.py
# if it is not dividable by 20, the custom variables in the storage will not be logged in the metrics.json file
val_period = 100

# smallest model, less AP, faster to train
network_type = "R_50_FPN_3x"

# faster training, but slightly less AP, way smaller model (.pth) file
# network_type = "R_101_FPN_3x"

# slowest training, but best AP
# network_type = "X_101_32x8d_FPN_3x"

model_dir = os.path.join(root_dir, "Models", network_type + "-" + json_pathname_extension + "2")
cfg_file = "COCO-Detection/faster_rcnn_" + network_type + ".yaml"

# if you already trained a model - link to its path with path
weight_file = "model_0000499.pth"
path_to_weight_file = os.path.join(model_dir, weight_file)

# to start training from scratch
# cfg, continue_training = setup_cfg(train_data_name, test_data_name, val_period, max_iter, len(type_of_annotation), model_dir, cfg_file)

# to continue training from weight file
cfg, continue_training = setup_cfg(train_data_name, test_data_name, val_period, max_iter, len(type_of_annotation), model_dir, cfg_file, path_to_weight_file)

# %%
# generate the coco annotations for the evaluator before the evaluator hook
COCOEvaluator(test_data_name, cfg, False, output_dir=cfg.OUTPUT_DIR) 
COCOEvaluator(val_data_name, cfg, False, output_dir=cfg.OUTPUT_DIR)

# %%
trainer = CustomTrainer(cfg, val_data_name, val_period)
trainer.resume_or_load(resume=continue_training)

# %%
trainer.train()

# %%