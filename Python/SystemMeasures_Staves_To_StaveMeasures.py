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
from MetricsVisualiser import MetricsVisualiser
from ImageDisplayer import ImageDisplayer

# %%
root_dir = "./../Data" # change this to download to a specific location on your pc
DataLoader().download_datasets(root_dir)
DataLoader().download_trained_models(root_dir)
DataLoader().generateAllJsonDataAnnotations(root_dir)

# %%
type_of_annotation = ["system_measures", "staves"]
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

muscima_data_name = "muscima"
muscima_metadata = registerDataset(muscima_data_name, muscima_data_name, muscima_data, type_of_annotation)

audioLabs_data_name = "audioLabs"
audioLabs_metadata = registerDataset(audioLabs_data_name, audioLabs_data_name, audioLabs_data, type_of_annotation)

# %%
def setup_cfg(musicma_data_name, audioLabs_data_name, num_classes, model_output_dir, cfg_file, existing_model_weight_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))

    cfg.OUTPUT_DIR = model_output_dir

    cfg.DATASETS.TRAIN = (musicma_data_name, audioLabs_data_name, )

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

model_dir = os.path.join(root_dir, "Models", network_type + "-" + json_pathname_extension)

cfg_file = "COCO-Detection/faster_rcnn_" + network_type + ".yaml"

weight_file = os.path.join(model_dir, "last_checkpoint")
last_checkpoint = open(weight_file, "r").read()

path_to_weight_file = os.path.join(model_dir, last_checkpoint) 

cfg = setup_cfg(muscima_data_name, audioLabs_data_name, len(type_of_annotation), model_dir, cfg_file, path_to_weight_file)

#%%
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
predictor = DefaultPredictor(cfg)

# %%
# fetch random raw prediction boxes:
for d in random.sample(muscima_data, 1):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    print(d["file_name"])
    print(outputs["instances"].pred_boxes)

    v = CustomVisualizer(im[:, :, ::-1], metadata=muscima_metadata, scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    ImageDisplayer().cv2_imshow(v.get_image()[:, :, ::-1])

# step by step process to calculate stave measures from the system measures and staves:

# %%  
# get all boxes for one image:
path_to_img = "./../Data\CVC_Muscima_Augmented\CVCMUSCIMA_MultiConditionAligned\staffline-y-variation-v1\w-25\p012.png"

ImageDisplayer().displaySpecificPredictData(predictor, muscima_data, muscima_metadata, path_to_img, [0])
ImageDisplayer().displaySpecificPredictData(predictor, muscima_data, muscima_metadata, path_to_img, [1])

im = cv2.imread(path_to_img)
outputs = predictor(im)
all_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
all_classes = outputs["instances"].pred_classes.cpu().numpy()
print(all_boxes)
print(all_classes)

# %%
# merge the boxes with the classes and sort by classes, because they dont have to be in order
np.set_printoptions(suppress=True) # or else it will print numbers such as 1243.68725586 like this 1.2436873e+03
box_classes = np.concatenate((all_boxes, np.array([all_classes]).T), axis=1)
sorted_box_classes = box_classes[box_classes[:,4].argsort()]
print(sorted_box_classes)

# %%
# get the boxes by classes
system_indices = np.where(sorted_box_classes == 0)[0]
stave_indices = np.where(sorted_box_classes == 1)[0]

system_boxes = sorted_box_classes[system_indices[0]:system_indices[-1]+1]
stave_boxes = sorted_box_classes[stave_indices[0]:stave_indices[-1]+1]
print("system_boxes\n", system_boxes, "\n\nstave_boxes\n", stave_boxes)

# %%
# sort the boxes by y values:
sorted_system_boxes = system_boxes[np.argsort(system_boxes[:, 1])]
sorted_stave_boxes = stave_boxes[np.argsort(stave_boxes[:, 1])]
print("sorted_system_boxes\n", sorted_system_boxes, "\n\nsorted_stave_boxes\n", sorted_stave_boxes)
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
            group_array.append([idx, box[0], box[1], box[2], box[3], box[4]])
            ref_y = box[1]
        else:
            group_array.append([idx, box[0], box[1], box[2], box[3], box[4]])

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
    bool_test = system[2]-10 < stave[1] and system[4]+10 > stave[3]
    return system[2]-10 < stave[1] and system[4]+10 > stave[3]

# %%
# now convert system measures and staves into stave measures for the image:
# first define the data frame that holds all our information:
data = {"Left" : [],
        "Top" : [],
        "Bottom" : [],
        "Right" : [],
        "Image" : []
        }
data_frame = pd.DataFrame(data, columns = ["Left", "Top", "Bottom", "Right", "Image"], dtype='int32')
data_frame.head()

# %%
# then run the algorithm:
system_measures = get_systems_with_index(sorted_system_boxes)
bboxes = sorted_stave_boxes

system_bounds = get_system_bounds(system_measures)
reference_systems = []

for stave in bboxes:
    for system in system_bounds:
        if is_in_system(stave, system): # does the stave fit into the system
            reference_systems = [k for i in system_measures for k in i if k[0] == system[0]] # all systems in system_measures with idx == system[0]
            # reference_systems = []
            # for i in system_measures:
            #     for k in i:
            #         if k[0] == system[0]:
            #             reference_systems.append(k)
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
                "Image" : ["imagename"]
                })

            data_frame = data_frame.append(data_row, ignore_index=True)
    
    reference_systems = []
    
# %%
print(data_frame)

# %%
from PIL import Image, ImageDraw
from IPython.display import display

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
img = draw_boxes(path_to_img, data_frame)
display(img)
# %%
# Now do it for all the images in the muscima dataset:

images_with_annotations = [i["file_name"].replace("\\", "/") for i in muscima_data]

path_to_musicma_folder = "./../Data/CVC_Muscima_Augmented/CVCMUSCIMA_MultiConditionAligned"

data = {"Left" : [],
        "Top" : [],
        "Bottom" : [],
        "Right" : [],
        "Image" : []
        }
data_frame = pd.DataFrame(data, columns = ["Left", "Top", "Bottom", "Right", "Image"], dtype='int32')

for augmentation_folders in tqdm(os.listdir(path_to_musicma_folder)):
    path_to_augmentation_folder = os.path.join(path_to_musicma_folder, augmentation_folders)
    for folder in os.listdir(path_to_augmentation_folder):
        path_to_images = os.path.join(path_to_augmentation_folder, folder)
        for image in os.listdir(path_to_images):
            img_path = os.path.join(path_to_images, image)

            # NOTE: not all images in the dataset have annotations,
            #       correct annotations are needed to calculate the COCO score
            #       thats why we only generate staveMeasures for the images with annotations
            if img_path.replace("\\", "/") in images_with_annotations:
                # get all boxes for the image:
                im = cv2.imread(img_path)
                outputs = predictor(im)
                all_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
                all_classes = outputs["instances"].pred_classes.cpu().numpy()

                # merge the boxes with the classes and sort by classes, because they dont have to be in order
                box_classes = np.concatenate((all_boxes, np.array([all_classes]).T), axis=1)
                sorted_box_classes = box_classes[box_classes[:,4].argsort()]

                # get the boxes by classes
                system_indices = np.where(sorted_box_classes == 0)[0]
                stave_indices = np.where(sorted_box_classes == 1)[0]

                system_boxes = sorted_box_classes[system_indices[0]:system_indices[-1]+1]
                stave_boxes = sorted_box_classes[stave_indices[0]:stave_indices[-1]+1]

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
                                "Image" : [img_path]
                                })

                            data_frame = data_frame.append(data_row, ignore_index=True)
                    
                    reference_systems = []


# %%
data_frame.to_csv("./../Data/Muscima_Generated_StaveMeasures.csv")

# %%

