# %%
from detectron2.config import get_cfg
from DataLoader import DataLoader
import os
from detectron2 import model_zoo
import torch
from detectron2.engine.defaults import DefaultPredictor
from PIL import Image
import numpy as np
from CustomVisualizer import CustomVisualizer
from ImageDisplayer import ImageDisplayer
import json
import cv2

root_dir = "./../Data" # change this to download to a specific location on your pc

#%%
DataLoader().download_trained_models(root_dir)

# %%
# to decide which data should be loaded use this:

# type_of_annotation = ["system_measures"]
# type_of_annotation = ["stave_measures"]
# type_of_annotation = ["staves"]

# type_of_annotation = ["system_measures", "staves"]
type_of_annotation = ["system_measures", "stave_measures", "staves"]

json_pathname_extension = "-".join(str(elem) for elem in type_of_annotation)

# %%
# to decide which network should be used to predict the data
network_type = "R_50_FPN_3x"
# network_type = "R_101_FPN_3x"
# network_type = "X_101_32x8d_FPN_3x"

# %%
# preparing the config variable and the predictor
def prepare_cfg_variables(model, category):
    model_dir = os.path.join(root_dir, "Models", model + "-" + category)
    cfg_file = "COCO-Detection/faster_rcnn_" + model + ".yaml"
    weight_file = os.path.join(model_dir, "last_checkpoint")
    last_checkpoint = open(weight_file, "r").read()
    path_to_weight_file = os.path.join(model_dir, last_checkpoint)
    return cfg_file, path_to_weight_file

def setup_cfg(num_classes, cfg_file, existing_model_weight_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))

    cfg.MODEL.WEIGHTS = existing_model_weight_path

    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    # set the testing threshold for this model. Model should be at least 20% confident detection is correct
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.2

    return cfg

cfg_file, path_to_weight_file = prepare_cfg_variables(network_type, json_pathname_extension) 

cfg = setup_cfg(len(type_of_annotation), cfg_file, path_to_weight_file)
predictor = DefaultPredictor(cfg)
    
# %%
# load your data
path_to_folder_with_images_to_predict = "../Data/CVC_Muscima_Augmented/CVCMUSCIMA_MultiConditionAligned/grayscale/w-02"
files_to_predict = []
for f in os.listdir(path_to_folder_with_images_to_predict):
    files_to_predict.append(path_to_folder_with_images_to_predict + "/" + f)
# %%
# predict your data
for f in files_to_predict:
    ImageDisplayer().displaySpecificPredictData(predictor, f)
# %%
# to create json files with raw data from predictions:
def generate_predictions_as_json(img_file_buffer, model, type_of_annotation, predictor, user_folder):
    if "system_measures-staves" == type_of_annotation:
        for img_file in img_file_buffer:
            json_dict = generate_JSON_multiple_category(img_file, predictor, 1)
            json_file_name = img_file.split("/")[-1].split(".")[0] + "-" + type_of_annotation + ".json"
            with open(os.path.join(user_folder, json_file_name), "w", encoding="utf8") as outfile:
                json.dump(json_dict, outfile, indent=4, ensure_ascii=False)
    elif "system_measures-stave_measures-staves" == type_of_annotation:
        for img_file in img_file_buffer:
            json_dict = generate_JSON_multiple_category(img_file, predictor, 2)
            json_file_name = img_file.split("/")[-1].split(".")[0] + "-" + type_of_annotation + ".json"
            with open(os.path.join(user_folder, json_file_name), "w", encoding="utf8") as outfile:
                json.dump(json_dict, outfile, indent=4, ensure_ascii=False)
    else:
        for img_file in img_file_buffer:
            json_dict = {}
            json_dict = generate_JSON_single_category(json_dict, img_file, predictor, type_of_annotation)
            json_file_name = img_file.split("/")[-1].split(".")[0] + "-" + type_of_annotation + ".json"
            with open(os.path.join(user_folder, json_file_name), "w", encoding="utf8") as outfile:
                json.dump(json_dict, outfile, indent=4, ensure_ascii=False)

def generate_JSON_single_category(json_dict, img_file, predictor, annotation_type):
    image = Image.open(img_file).convert("RGB")
    im = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    outputs = predictor(im)
    all_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy() # left, top, right, bottom

    json_dict["width"] = image.width
    json_dict["height"] = image.height

    measures = []
    for box in all_boxes:
        annotation = {}
        annotation["left"] = int(box[0].item())
        annotation["top"] = int(box[1].item())
        annotation["right"] = int(box[2].item())
        annotation["bottom"] = int(box[3].item())
        measures.append(annotation)

    json_dict[annotation_type] = measures

    return json_dict

def generate_JSON_multiple_category(img_file, predictor, category):
    image = Image.open(img_file).convert("RGB")
    im = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    outputs = predictor(im)

    all_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy() # left, top, right, bottom
    all_classes = outputs["instances"].pred_classes.cpu().numpy()

    # merge the boxes with the classes and sort by classes, because they dont have to be in order
    box_classes = np.concatenate((all_boxes, np.array([all_classes]).T), axis=1)
    sorted_box_classes = box_classes[box_classes[:,4].argsort()]

    # get the boxes by classes but without the class column
    system_boxes = None
    stave_measure_boxes = None
    stave_boxes = None

    if category == 1:
        system_indices = np.where(sorted_box_classes == 0)[0]
        stave_indices = np.where(sorted_box_classes == 1)[0]

        system_boxes = np.delete(sorted_box_classes[system_indices[0]:system_indices[-1]+1], 4, 1)
        stave_boxes = np.delete(sorted_box_classes[stave_indices[0]:stave_indices[-1]+1], 4, 1)
    elif category == 2:
        system_indices = np.where(sorted_box_classes == 0)[0]
        stave_measure_indices = np.where(sorted_box_classes == 1)[0]
        stave_indices = np.where(sorted_box_classes == 2)[0]

        system_boxes = np.delete(sorted_box_classes[system_indices[0]:system_indices[-1]+1], 4, 1)
        stave_measure_boxes = np.delete(sorted_box_classes[stave_measure_indices[0]:stave_measure_indices[-1]+1], 4, 1)
        stave_boxes = np.delete(sorted_box_classes[stave_indices[0]:stave_indices[-1]+1], 4, 1)

    json_dict = {}
    json_dict["width"] = image.width
    json_dict["height"] = image.height

    measures = []
    for box in system_boxes:
        annotation = {}
        annotation["left"] = int(box[0].item())
        annotation["top"] = int(box[1].item())
        annotation["right"] = int(box[2].item())
        annotation["bottom"] = int(box[3].item())
        measures.append(annotation)
    json_dict["system_measures"] = measures

    if category == 2:
        measures = []
        for box in stave_measure_boxes:
            annotation = {}
            annotation["left"] = int(box[0].item())
            annotation["top"] = int(box[1].item())
            annotation["right"] = int(box[2].item())
            annotation["bottom"] = int(box[3].item())
            measures.append(annotation)
        json_dict["stave_measures"] = measures

    measures = []
    for box in stave_boxes:
        annotation = {}
        annotation["left"] = int(box[0].item())
        annotation["top"] = int(box[1].item())
        annotation["right"] = int(box[2].item())
        annotation["bottom"] = int(box[3].item())
        measures.append(annotation)
    json_dict["staves"] = measures

    return json_dict

# %%
generate_predictions_as_json(files_to_predict, network_type, json_pathname_extension, predictor, "../CustomDataFolder")

# %%