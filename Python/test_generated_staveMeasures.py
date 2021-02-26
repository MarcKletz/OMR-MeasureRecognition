# %%
import os, sys
import cv2
import numpy as np
import pandas as pd

import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog

from DataLoader import DataLoader
from ImageDisplayer import ImageDisplayer

from PIL import Image, ImageDraw
from IPython.display import display

# %%
root_dir = "./../Data" # change this to download to a specific location on your pc

# %%
network_type = "R_50"
# network_type = "R_101"
# network_type = "X_101"

network_used = "SingleNetwork"
# TODO: test twoNN results - not possible right now
# network_used = "TwoNN_SystemAndStaves"

data_frame = pd.read_csv(os.path.join(root_dir, network_type + "_" + network_used + "_StaveMeasures.csv"))
data_frame.head()
# %%
folder_prefix = "/CVC_Muscima_Augmented/CVCMUSCIMA_MultiConditionAligned"
# image = "\\binary\\w-02\\p017.png"
image = "\\interrupted\\w-49\\p003.png"

image_to_display = root_dir + folder_prefix + image
image_to_display = image_to_display.replace("\\", "/")
print(image_to_display)

image_df = data_frame.loc[data_frame["Image"] == image_to_display]
image_df.head()

# %%
# show original image
im = cv2.imread(image_to_display)
ImageDisplayer().cv2_imshow(im)

# %%
# show predicted boxes:
type_of_annotation = ["system_measures", "staves"]
json_pathname_extension = "-".join(str(elem) for elem in type_of_annotation)

# %%
json_path = os.path.join(root_dir, "CVC_muscima_" + json_pathname_extension + ".json")
muscima_data = DataLoader().load_from_json(json_path)

def registerDataset(data_name, d, data, classes):
    DatasetCatalog.register(data_name, lambda d=d: data)
    MetadataCatalog.get(data_name).set(thing_classes=classes)

    return MetadataCatalog.get(data_name)

muscima_data_name = "muscima"
muscima_metadata = registerDataset(muscima_data_name, muscima_data_name, muscima_data, type_of_annotation)
# %%
def setup_cfg(num_classes, model_output_dir, cfg_file, existing_model_weight_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))

    cfg.OUTPUT_DIR = model_output_dir

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

network_type = "R_50_FPN_3x"
# network_type = "R_101_FPN_3x"
# network_type = "X_101_32x8d_FPN_3x"

model_dir = os.path.join(root_dir, "Models", network_type + "-" + json_pathname_extension)

cfg_file = "COCO-Detection/faster_rcnn_" + network_type + ".yaml"

weight_file = os.path.join(model_dir, "last_checkpoint")
last_checkpoint = open(weight_file, "r").read()

path_to_weight_file = os.path.join(model_dir, last_checkpoint) 

cfg = setup_cfg(len(type_of_annotation), model_dir, cfg_file, path_to_weight_file)
predictor = DefaultPredictor(cfg)

ImageDisplayer().displaySpecificPredictData(predictor, image_to_display, [0])
ImageDisplayer().displaySpecificPredictData(predictor, image_to_display, [1])

# %%
def draw_boxes(image_path, df_page):
    print(image_path)
    img = Image.open(image_path).convert('RGB')
    bboxes = df_page[['Left', 'Top', 'Bottom', 'Right']].values
    draw = ImageDraw.Draw(img)
    
    for left, top, bottom, right in bboxes:
        points = (left, top), (right, top), (right, bottom), (left, bottom), (left, top)
        draw.line(points, fill='red', width=5)
    
    return img
# %%
# display the generated boxes
draw_boxes(image_to_display, image_df)

# %%
