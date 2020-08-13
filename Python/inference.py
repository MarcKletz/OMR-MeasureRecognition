# %%
# import some common libraries
import torch, torchvision
import numpy as np
import pandas as pd
import cv2
import os
import random
import json
from tqdm import tqdm
from matplotlib import pyplot as plt

# import some detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor

# importing custom classes from the repo
from DataLoader import DataLoader
from ImageDisplayer import ImageDisplayer
from CustomVisualiser import CustomVisualizer

# %%
root_dir = "./../Data" # change this to download to a specific location on your pc

#%%
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
    
muscima_metadata = registerDataset("muscima", "muscima", muscima_data, type_of_annotation)
audioLabs_metadata = registerDataset("audioLabs", "audioLabs", audioLabs_data, type_of_annotation)

# %%
ImageDisplayer().displayRandomSampleData(muscima_data, muscima_metadata, 1)

# %%
ImageDisplayer().displayRandomSampleData(audioLabs_data, audioLabs_metadata, 1)

# %%
def setup_cfg(num_classes, cfg_file, existing_model_weight_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))

    cfg.MODEL.WEIGHTS = existing_model_weight_path

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set the testing threshold for this model. Model should be at least 20% confident detection is correct
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.2

    # Set seed to negative to fully randomize everything.
    # Set seed to positive to use a fixed seed. Note that a fixed seed increases
    # reproducibility but does not guarantee fully deterministic behavior.
    # Disabling all parallelism further increases reproducibility.
    cfg.SEED = 1

    return cfg

# %%
network_type = "R_50_FPN_3x"
# network_type = "R_101_FPN_3x"

model_dir = os.path.join(root_dir, "Models", network_type + "-" + json_pathname_extension)

cfg_file = "COCO-Detection/faster_rcnn_" + network_type + ".yaml"

weight_file = "final_" + json_pathname_extension + "_model.pth"
# weight_file = "model_0000599.pth"

path_to_weight_file = os.path.join(model_dir, weight_file) 

#%%
cfg = setup_cfg(len(type_of_annotation), cfg_file, path_to_weight_file)
predictor = DefaultPredictor(cfg)

# %%
ImageDisplayer().displayRandomPredictData(predictor, muscima_data, muscima_metadata, 5)

# %%
ImageDisplayer().displayRandomPredictData(predictor, audioLabs_data, audioLabs_metadata, 5)

# %%
