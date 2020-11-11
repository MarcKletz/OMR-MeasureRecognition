#%%
import os, sys
import random
import cv2
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import detectron2
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer

from CustomVisualizer import CustomVisualizer
from DataLoader import DataLoader
from MetricsVisualizer import MetricsVisualizer
from ImageDisplayer import ImageDisplayer

# %%
root_dir = "./../Data" # change this to download to a specific location on your pc
DataLoader().download_datasets(root_dir)
DataLoader().download_trained_models(root_dir)
DataLoader().generateAllJsonDataAnnotations(root_dir)

# %%
annotation_type = "system_measures"
json_path = os.path.join(root_dir, "CVC_muscima_" + annotation_type + ".json")
muscima_data = DataLoader().load_from_json(json_path)

json_path = os.path.join(root_dir, "AudioLabs_" + annotation_type + ".json")
audioLabs_data = DataLoader().load_from_json(json_path)

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
metadata = registerDataset(train_data_name, train_data_name, train_data, annotation_type)

test_data_name = "test"
registerDataset(test_data_name, test_data_name, test_data, annotation_type)

# %%
def setup_cfg(data_name, num_classes, model_output_dir, cfg_file, existing_model_weight_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))

    cfg.OUTPUT_DIR = model_output_dir

    cfg.DATASETS.TRAIN = (data_name, )

    cfg.DATALOADER.NUM_WORKERS = 0 # Number of data loading threads

    cfg.MODEL.WEIGHTS = existing_model_weight_path

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # 128 faster, and good enough for toy dataset (default: 512)
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
# network_type = "X_101_32x8d_FPN_3x"

# %%
annotation_type = "system_measures"
model_dir = os.path.join(root_dir, "Models", network_type + "-" + annotation_type)
cfg_file = "COCO-Detection/faster_rcnn_" + network_type + ".yaml"
weight_file = os.path.join(model_dir, "last_checkpoint")
last_checkpoint = open(weight_file, "r").read()
path_to_weight_file = os.path.join(model_dir, last_checkpoint) 
system_cfg = setup_cfg(train_data_name, 1, model_dir, cfg_file, path_to_weight_file)

system_trainer = DefaultTrainer(system_cfg)
system_trainer.resume_or_load(resume=True)
system_predictor = DefaultPredictor(system_cfg)

annotation_type = "staves"
model_dir = os.path.join(root_dir, "Models", network_type + "-" + annotation_type)
cfg_file = "COCO-Detection/faster_rcnn_" + network_type + ".yaml"
weight_file = os.path.join(model_dir, "last_checkpoint")
last_checkpoint = open(weight_file, "r").read()
path_to_weight_file = os.path.join(model_dir, last_checkpoint) 
stave_cfg = setup_cfg(train_data_name, 1, model_dir, cfg_file, path_to_weight_file)

stave_trainer = DefaultTrainer(stave_cfg)
stave_trainer.resume_or_load(resume=True)
stave_predictor = DefaultPredictor(stave_cfg)

# %%
def sort_by_x(elem):
    return elem[1]

# defining some helper functions used in the algorithm
def get_systems_with_index(system_boxes):
    array = []
    threshold = 40
    ref_y = 0
    group_array = []

    idx = 0
    for box in system_boxes:
        if box[1] > ref_y + threshold:
            if len(group_array) != 0:
                group_array.sort(key=sort_by_x)
                array.append(group_array)
                idx += 1
            group_array = []
            group_array.append([idx, box[0], box[1], box[2], box[3]])
            ref_y = box[1]
        else:
            group_array.append([idx, box[0], box[1], box[2], box[3]])

    group_array.sort(key=sort_by_x)
    array.append(group_array)

    return array

def get_system_bounds(system_measures):
    system_bounds = []
    for systems in system_measures:
        left = math.inf
        top = math.inf
        right = 0
        bottom = 0
        system_idx = -1
        for system in systems:
            system_idx = system[0]
            if system[1] < left:
                left = system[1]
            if system[2] < top:
                top = system[2]
            if system[3] > right:
                right = system[3]
            if system[4] > bottom:
                bottom = system[4]
        system_bounds.append([system_idx, left, top, right, bottom])

    return system_bounds

def is_in_system(stave, system):
    # stave in format left, top, right, bottom, idx
    # system in format: idx, left , top, right, bottom
    threshold = 10
    bool_test = system[2]-threshold < stave[1] and system[4]+threshold > stave[3]
    return system[2]-threshold < stave[1] and system[4]+threshold > stave[3]

# %%
def generate_stave_annotations(anno_file_paths):
    data = {"Left" : [],
        "Top" : [],
        "Bottom" : [],
        "Right" : [],
        "Image" : []
        }
    data_frame = pd.DataFrame(data, columns = ["Left", "Top", "Bottom", "Right", "Image"], dtype='int32')

    for anno_img in tqdm(anno_file_paths):
        # get all boxes for the image:
        im = cv2.imread(anno_img)
        outputs = system_predictor(im)
        system_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

        outputs = stave_predictor(im)
        stave_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

        # there are images in the Measure bounding box annotations v2 with no annotations
        # nor any music notes, empty pages or pages with text
        # the network does not always detect system measures or staves (which is correct in such a case)
        # TODO: (?) it might be beneficial to remove such images beforehand (?)
        if len(system_boxes) == 0 or len(stave_boxes) == 0:
            continue

        # sort the boxes by y values:
        sorted_system_boxes = system_boxes[np.argsort(system_boxes[:, 1])]
        sorted_stave_boxes = stave_boxes[np.argsort(stave_boxes[:, 1])]

        # then run the algorithm:
        system_measures = get_systems_with_index(sorted_system_boxes)
        bboxes = sorted_stave_boxes

        system_bounds = get_system_bounds(system_measures)
        reference_systems = []

        for stave in bboxes:
            for system in system_bounds:
                if is_in_system(stave, system): # does the stave fit into the system
                    # all systems in system_measures with idx == system[0]
                    reference_systems = [k for i in system_measures for k in i if k[0] == system[0]]
                    break

            # split the stave with the reference system annotations by
            # y_top and y_bottom of the stave but the x_left and x_right of the reference system
            if len(reference_systems) != 0:
                for system_measure in reference_systems:
                    left = system_measure[1]
                    top = stave[1]
                    bottom = stave[3]
                    right = system_measure[3]

                    data_row = pd.DataFrame({
                        "Left" : [left],
                        "Top" : [top],
                        "Bottom" : [bottom],
                        "Right" : [right],
                        "Image" : [anno_img]
                        })

                    data_frame = data_frame.append(data_row, ignore_index=True)
            
            reference_systems = []
    
    return data_frame

# %%
anno_paths = [i["file_name"].replace("\\", "/") for i in test_data]

data_frame = generate_stave_annotations(anno_paths)

# %%
network_name_split = network_type.split("_")
network_used = network_name_split[0] + "_" + network_name_split[1]
data_frame.to_csv("./../Data/" + network_used + "_TwoNN_SystemAndStaves_StaveMeasures.csv")

# %%
