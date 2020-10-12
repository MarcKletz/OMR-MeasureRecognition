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
import sys
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
from CustomVisualizer import CustomVisualizer
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
count = 0
for d in muscima_data:
    count += len(d["annotations"])
count
# %%
count = 0
for d in audioLabs_data:
    count += len(d["annotations"])
count

# %%
