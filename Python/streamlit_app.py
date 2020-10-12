import io
from PIL import Image
import streamlit as st

# import some common libraries
import torch, torchvision
import numpy as np
import pandas as pd
import cv2
import os

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
                handle_model_ensemble(img_file_buffer, model, display_original_image)
            else:
                handle_standard_prediction(img_file_buffer, model, display_original_image, type_of_annotation)
    elif what_do == "Download predictions":
        display_metrics(model, type_of_annotation, with_visualizer=False)
        
        # if st.sidebar.button("Download predicted annotations as JSON."):
        #     download_predictions_as_json(img_file_buffer, model, type_of_annotation)

        """
        One JSON file will be generated for each image.  
        The file name will be the same as the image name.  
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
        \t\t\t\"height\": <bounding_box_1_height>
        \t\t\t\"width\": <bounding_box_1_width>
        \t\t},
        \t\t{
        \t\t\t\"left\": <bounding_box_2_left>
        \t\t\t\"top\": <bounding_box_2_top>
        \t\t\t\"height\": <bounding_box_2_height>
        \t\t\t\"width\": <bounding_box_2_width>
        \t\t}
        \t],
        \t\"<category_2>\": [
        \t\t{
        \t\t\t\"left\": <bounding_box_1_left>
        \t\t\t\"top\": <bounding_box_1_top>
        \t\t\t\"height\": <bounding_box_1_height>
        \t\t\t\"width\": <bounding_box_1_width>
        \t\t},
        \t\t{
        \t\t\t\"left\": <bounding_box_2_left>
        \t\t\t\"top\": <bounding_box_2_top>
        \t\t\t\"height\": <bounding_box_2_height>
        \t\t\t\"width\": <bounding_box_2_width>
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

def handle_model_ensemble(img_file_buffer, model, display_original_image):
    for c in all_classes:
        model_dir = os.path.join(root_dir, "Models", model + "-" + c)
        cfg_file = "COCO-Detection/faster_rcnn_" + model + ".yaml"
        weight_file = os.path.join(model_dir, "last_checkpoint")
        last_checkpoint = open(weight_file, "r").read()
        path_to_weight_file = os.path.join(model_dir, last_checkpoint) 

        cfg = setup_cfg(1, cfg_file, path_to_weight_file)
        predictor = DefaultPredictor(cfg)

        for img_file in img_file_buffer:
            predict_image(predictor, img_file, display_original_image, custom_message=c)

def handle_standard_prediction(img_file_buffer, model, display_original_image, type_of_annotation):
    model_dir = os.path.join(root_dir, "Models", model + "-" + type_of_annotation)
    cfg_file = "COCO-Detection/faster_rcnn_" + model + ".yaml"
    weight_file = os.path.join(model_dir, "last_checkpoint")
    last_checkpoint = open(weight_file, "r").read()
    path_to_weight_file = os.path.join(model_dir, last_checkpoint) 
    display_multiple_classes = False

    which_classes = []
    if type_of_annotation == "system_measures-stave_measures-staves":
        cfg = setup_cfg(3, cfg_file, path_to_weight_file)
        display_multiple_classes = True
        which_classes = [all_classes.index(x) for x in type_of_annotation.split("-") if x in all_classes]
    else:
        cfg = setup_cfg(1, cfg_file, path_to_weight_file)
    predictor = DefaultPredictor(cfg)

    for img_file in img_file_buffer:
        predict_image(predictor, img_file, display_original_image, display_multiple_classes, which_classes)

def predict_image(predictor, img_file, display_original_image, display_multiple_classes=False, which_classes=[], custom_message=""):
    image = Image.open(img_file).convert("RGB")
    if display_original_image:
        st.image(image, "Your uploaded image", use_column_width=True)

    im = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    outputs = predictor(im)

    if display_multiple_classes:
        for c in which_classes:
            v = CustomVisualizer(im[:, :, ::-1], scale=1)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"), [c])
            st.image(v.get_image()[:, :, ::-1], "The " + all_classes[c] + " predictions of the network", use_column_width=True)

    v = CustomVisualizer(im[:, :, ::-1], scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    st.image(v.get_image()[:, :, ::-1], "All " + custom_message + " predictions of the network", use_column_width=True)

def setup_cfg(num_classes, cfg_file, existing_model_weight_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))

    cfg.MODEL.WEIGHTS = existing_model_weight_path

    if not torch.cuda.is_available():
        st.write("NO NVIDIA GPU FOUND - fallback to CPU")
        cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    # set the testing threshold for this model. Model should be at least 20% confident detection is correct
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.2

    return cfg

# def download_predictions_as_json(img_file_buffer, model, type_of_annotation):
#     if type_of_annotation == "model ensemble":
#         st.write("Later")
#     elif type_of_annotation == "system_measures-stave_measures-staves":
#         st.write("Later")
#     else:
#         model_dir = os.path.join(root_dir, "Models", model + "-" + type_of_annotation)
#         cfg_file = "COCO-Detection/faster_rcnn_" + model + ".yaml"
#         weight_file = os.path.join(model_dir, "last_checkpoint")
#         last_checkpoint = open(weight_file, "r").read()
#         path_to_weight_file = os.path.join(model_dir, last_checkpoint) 
#         cfg = setup_cfg(1, cfg_file, path_to_weight_file)




        # json_dict = {}
        # json_dict["width"] = 
        # json_dict["height"] = 
        # system_measures = []
        # measure_1 = {}
        # json_dict["system_measures"] = system_measures

if __name__ == "__main__":
    main()