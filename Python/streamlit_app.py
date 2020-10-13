import pathlib
import random
import string
import shutil

import io
from PIL import Image
import streamlit as st

# import some common libraries
import torch, torchvision
import numpy as np
import pandas as pd
import cv2
import os
import json

# import some detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor

from CustomVisualizer import CustomVisualizer
from DataLoader import DataLoader
from MetricsVisualizer import MetricsVisualizer

"""
# OMR Measure Recognition
by Marc Kletz
"""

root_dir = "./Data"
all_classes = ["system_measures", "stave_measures", "staves"]

# HACK This only works when we've installed streamlit with pipenv, so the
# permissions during install are the same as the running process
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static' / 'downloads'

def main():
    DataLoader().download_trained_models(root_dir)

    st.sidebar.title("What to do")

    what_do = st.sidebar.selectbox("Select what to do", ["Show metrics", "Inference", "Download predictions"])
    model = st.sidebar.selectbox("Choose a model", ["R_50_FPN_3x", "R_101_FPN_3x"])
    type_of_annotation = st.sidebar.selectbox("Choose the type of annotation the model looks for",
        ["staves", "system_measures", "stave_measures", "system_measures-staves", "system_measures-stave_measures-staves", "model ensemble"]) 

    img_file_buffer = st.file_uploader("Upload image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if what_do == "Show metrics":
        display_metrics(model, type_of_annotation)
    elif what_do == "Inference":
        display_original_image = st.sidebar.checkbox("Display the original image(s)", False)

        if(len(img_file_buffer) > 0):
            if type_of_annotation == "model ensemble":
                handle_prediction(img_file_buffer, model, display_original_image, all_classes)
            else:
                handle_prediction(img_file_buffer, model, display_original_image, [type_of_annotation])
    elif what_do == "Download predictions":
        display_metrics(model, type_of_annotation, with_visualizer=False)

        user_folder_input = st.sidebar.text_input("Folder name (where your generated JSON will be written to - if no value is given, we generate a random string)", max_chars=25)
        
        if st.sidebar.button("Generate predicted annotations as JSON."):
            if user_folder_input == "":
                st.write("FOLDER NAME IS EMPTY - GENERATING RANDOM STRING")
                user_folder_input = ''.join(random.choice(string.ascii_lowercase) for i in range(25))

            user_folder = (STREAMLIT_STATIC_PATH / user_folder_input)
            
            if os.path.exists(user_folder):
                st.write("FOLDER PATH ALREADY EXISTS - WE WILL NOW CLEAR THIS FOLDER")
                for f in os.listdir(user_folder):
                    os.remove(os.path.join(user_folder, f))
            else:
                os.mkdir(user_folder)

            if type_of_annotation == "model ensemble":
                generate_predictions_as_json(img_file_buffer, model, all_classes, user_folder_input, user_folder)
            else:
                generate_predictions_as_json(img_file_buffer, model, [type_of_annotation], user_folder_input, user_folder)

        """
        ..........................................................................................................  
        One JSON file will be generated for each image.  
        The file name will have the same as the image name with the category appended.  
        Example: File_1-staves.json\n
        Categories are the type of annotation selected: staves or sytem measures or stave measures or one of the joined categories.  
        Multiple categories will still be annotated in the same JSON as seen below.  
        The contents of the file will be in the following structure:  
        ```
        {
        \t\"width\": <image_width>,
        \t\"height\": <image_height>,
        \t\"<category_1>\": [
        \t\t{
        \t\t\t\"left\": <bounding_box_1_left>
        \t\t\t\"top\": <bounding_box_1_top>
        \t\t\t\"right\": <bounding_box_1_right>
        \t\t\t\"bottom\": <bounding_box_1_bottom>
        \t\t},
        \t\t{
        \t\t\t\"left\": <bounding_box_2_left>
        \t\t\t\"top\": <bounding_box_2_top>
        \t\t\t\"right\": <bounding_box_2_right>
        \t\t\t\"bottom\": <bounding_box_2_bottom>
        \t\t}
        \t],
        \t\"<category_2>\": [
        \t\t{
        \t\t\t\"left\": <bounding_box_1_left>
        \t\t\t\"top\": <bounding_box_1_top>
        \t\t\t\"right\": <bounding_box_1_right>
        \t\t\t\"bottom\": <bounding_box_1_bottom>
        \t\t},
        \t\t{
        \t\t\t\"left\": <bounding_box_2_left>
        \t\t\t\"top\": <bounding_box_2_top>
        \t\t\t\"right\": <bounding_box_2_right>
        \t\t\t\"bottom\": <bounding_box_2_bottom>
        \t\t}
        \t]
        }
        ```
        """

def display_metrics(model, type_of_annotation, with_visualizer=True):
    df = pd.DataFrame(columns=["Model Name", "Iterations", "mAP", "AP75", "AP50"])
    df.style.format({"E" : "{:.3%}"})
    if model == "R_50_FPN_3x":
        df = df.append({"Model Name" : "System measures", "Iterations" : 16500, "mAP" : 96.547, "AP75" : 98.935, "AP50" : 98.970, "system measures mAP" : "XX", "staves mAP" : "XX", "stave measures mAP" : "XX"}, ignore_index=True)
        df = df.append({"Model Name" : "Stave measures", "Iterations" : 16200, "mAP" : 88.168, "AP75" : 96.860, "AP50" : 98.013, "system measures mAP" : "XX", "staves mAP" : "XX", "stave measures mAP" : "XX"}, ignore_index=True)
        df = df.append({"Model Name" : "Staves", "Iterations" : 10800, "mAP" : 93.596, "AP75" : 98.987, "AP50" : 100.00, "system measures mAP" : "XX", "staves mAP" : "XX", "stave measures mAP" : "XX"}, ignore_index=True)
        df = df.append({"Model Name" : "System measures and Staves", "Iterations" : 15000, "mAP" : 90.792, "AP75" : 96.481, "AP50" : 96.500, "system measures mAP" : 95.532, "staves mAP" : 86.053, "stave measures mAP" : "XX"}, ignore_index=True)
        df = df.append({"Model Name" : "System measures, Stave measures and Staves", "Iterations" : 4500, "mAP" : 77.779, "AP75" : 86.884, "AP50" : 87.404, "system measures mAP" : 90.010, "staves mAP" : 78.622, "stave measures mAP" : 64.706}, ignore_index=True)
    elif model == "R_101_FPN_3x":
        df = df.append({"Model Name" : "System measures", "Iterations" : 19500, "mAP" : 97.112, "AP75" : 98.928, "AP50" : 98.949, "system measures mAP" : "XX", "staves mAP" : "XX", "stave measures mAP" : "XX"}, ignore_index=True)
        df = df.append({"Model Name" : "Stave measures", "Iterations" : 17100, "mAP" : 89.254, "AP75" : 97.903, "AP50" : 98.018, "system measures mAP" : "XX", "staves mAP" : "XX", "stave measures mAP" : "XX"}, ignore_index=True)
        df = df.append({"Model Name" : "Staves", "Iterations" : 18000, "mAP" : 94.004, "AP75" : 99.010, "AP50" : 100.00, "system measures mAP" : "XX", "staves mAP" : "XX", "stave measures mAP" : "XX"}, ignore_index=True)
        df = df.append({"Model Name" : "System measures and Staves", "Iterations" : 18300, "mAP" : 91.301, "AP75" : 96.478, "AP50" : 96.498, "system measures mAP" : 95.768, "staves mAP" : 86.834, "stave measures mAP" : "XX"}, ignore_index=True)
        df = df.append({"Model Name" : "System measures, Stave measures and Staves", "Iterations" : 2700, "mAP" : 77.829, "AP75" : 88.697, "AP50" : 89.366, "system measures mAP" : 85.383, "staves mAP" : 79.779, "stave measures mAP" : 68.324}, ignore_index=True)
    
    if type_of_annotation == "system_measures":
        st.table(df.loc[0:0].set_index("Model Name"))
    elif type_of_annotation == "stave_measures":
        st.table(df.loc[1:1].set_index("Model Name"))
    elif type_of_annotation == "staves":
        st.table(df.loc[2:2].set_index("Model Name"))
    elif type_of_annotation == "system_measures-staves":
        st.table(df.loc[3:3].set_index("Model Name"))
    elif type_of_annotation == "system_measures-stave_measures-staves":
        st.table(df.loc[4:4].set_index("Model Name"))
    elif type_of_annotation == "model ensemble":
        st.table(df.loc[0:2].set_index("Model Name"))
    
    if(with_visualizer):
        if type_of_annotation == "model ensemble":
            for c in all_classes:
                st.markdown("# " + c)
                MetricsVisualizer().visualizeMetrics(root_dir, model, [c])
        else:
            metrics_type_annotations = [x for x in type_of_annotation.split("-") if x in all_classes]
            st.markdown("# " + type_of_annotation)
            MetricsVisualizer().visualizeMetrics(root_dir, model, metrics_type_annotations)

def handle_prediction(img_file_buffer, model, display_original_image, type_of_annotation):
    for category in type_of_annotation:
        cfg_file, path_to_weight_file = prepare_cfg_variables(model, category) 

        nr_of_classes = 1
        which_classes = []
        display_multiple_classes = False
        if category == "system_measures-staves":
            nr_of_classes = 2
            which_classes = [0,1]
            display_multiple_classes = True
        elif category == "system_measures-stave_measures-staves":
            nr_of_classes = 3
            which_classes = [0,1,2]
            display_multiple_classes = True

        cfg = setup_cfg(nr_of_classes, cfg_file, path_to_weight_file)
        predictor = DefaultPredictor(cfg)

        for img_file in img_file_buffer:
            predict_image(predictor, img_file, display_original_image, display_multiple_classes, which_classes)

def prepare_cfg_variables(model, category):
    model_dir = os.path.join(root_dir, "Models", model + "-" + category)
    cfg_file = "COCO-Detection/faster_rcnn_" + model + ".yaml"
    weight_file = os.path.join(model_dir, "last_checkpoint")
    last_checkpoint = open(weight_file, "r").read()
    path_to_weight_file = os.path.join(model_dir, last_checkpoint)
    return cfg_file, path_to_weight_file

def predict_image(predictor, img_file, display_original_image, display_multiple_classes, which_classes):
    image = Image.open(img_file).convert("RGB")
    if display_original_image:
        st.image(image, "Your uploaded image", use_column_width=True)

    im = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    outputs = predictor(im)

    if display_multiple_classes:
        for c in which_classes:
            v = CustomVisualizer(im[:, :, ::-1], scale=1)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"), [c])
            st.image(v.get_image()[:, :, ::-1], use_column_width=True)

    v = CustomVisualizer(im[:, :, ::-1], scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    st.image(v.get_image()[:, :, ::-1], use_column_width=True)

def setup_cfg(num_classes, cfg_file, existing_model_weight_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))

    cfg.MODEL.WEIGHTS = existing_model_weight_path

    if not torch.cuda.is_available():
        st.write("NO NVIDIA GPU FOUND - fallback to CPU")
        st.write("This will be rather slow!")
        cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    # set the testing threshold for this model. Model should be at least 20% confident detection is correct
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.2

    return cfg

def generate_predictions_as_json(img_file_buffer, model, type_of_annotation, user_folder_input, user_folder):
    if "system_measures-staves" in type_of_annotation:
        cfg_file, path_to_weight_file = prepare_cfg_variables(model, type_of_annotation[0])
        cfg = setup_cfg(2, cfg_file, path_to_weight_file)
        predictor = DefaultPredictor(cfg)

        for img_file in img_file_buffer:
            json_dict = generate_JSON_multiple_category(img_file, predictor, 1)
            json_file_name = img_file.name.split(".")[0] + "-" + type_of_annotation[0] + ".json"
            with open(os.path.join(user_folder, json_file_name), "w", encoding="utf8") as outfile:
                json.dump(json_dict, outfile, indent=4, ensure_ascii=False)
    elif "system_measures-stave_measures-staves" in type_of_annotation:
        cfg_file, path_to_weight_file = prepare_cfg_variables(model, type_of_annotation[0])
        cfg = setup_cfg(3, cfg_file, path_to_weight_file)
        predictor = DefaultPredictor(cfg)

        for img_file in img_file_buffer:
            json_dict = generate_JSON_multiple_category(img_file, predictor, 2)
            json_file_name = img_file.name.split(".")[0] + "-" + type_of_annotation[0] + ".json"
            with open(os.path.join(user_folder, json_file_name), "w", encoding="utf8") as outfile:
                json.dump(json_dict, outfile, indent=4, ensure_ascii=False)
    else:
        for img_file in img_file_buffer:
            json_dict = {}
            for category in type_of_annotation:
                cfg_file, path_to_weight_file = prepare_cfg_variables(model, category) 
                cfg = setup_cfg(1, cfg_file, path_to_weight_file)
                predictor = DefaultPredictor(cfg)

                json_dict = generate_JSON_single_category(json_dict, img_file, predictor, category)
            json_file_name = img_file.name.split(".")[0] + "-" + type_of_annotation[0] + ".json"
            with open(os.path.join(user_folder, json_file_name), "w", encoding="utf8") as outfile:
                json.dump(json_dict, outfile, indent=4, ensure_ascii=False)

    shutil.make_archive(user_folder, "zip", user_folder)
    st.markdown("Download [" + user_folder_input + ".zip](downloads/" + user_folder_input + ".zip)")


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


if __name__ == "__main__":
    main()