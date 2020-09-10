import ctypes
libgcc_s = ctypes.CDLL("/usr/lib/libgcc_s.so.1")

import cv2

import io
from PIL import Image
import streamlit as st

# import some common libraries
import torch, torchvision
import numpy as np
import pandas as pd
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
from MetricsVisualiser import MetricsVisualiser

"""
# OMR Measure Recognition
by Marc Kletz
"""

root_dir = "./Data"
all_classes = ["system_measures", "stave_measures", "staves"]

def main():
    DataLoader().download_trained_models(root_dir)

    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.title("What to do")

    what_do = st.sidebar.selectbox("Select what to do", ["Show metrics", "Inference"])
    model = st.sidebar.selectbox("Choose a model", ["R_50_FPN_3x", "R_101_FPN_3x"])
    type_of_annotation = st.sidebar.selectbox("Choose the type of annotation the model looks for",
        ["staves", "system_measures", "stave_measures", "system_measures-stave_measures-staves", "model ensemble"])

    if what_do == "Show metrics":
        display_metrics(model, type_of_annotation)
        
    elif what_do == "Inference":
        display_original_image = st.sidebar.checkbox("Display the original image(s)", False)

        if type_of_annotation == "model ensemble":
            handle_model_ensemble(model, display_original_image)
        else:
            handle_standard_prediction(model, display_original_image, type_of_annotation)

def display_metrics(model, type_of_annotation):
    df = pd.DataFrame(columns=["Model Name", "Iterations", "mAP", "AP75", "AP50"])
    df.style.format({"E" : "{:.3%}"})
    if model == "R_50_FPN_3x":
        df = df.append({"Model Name" : "Staves", "Iterations" : 5700, "mAP" : 93.173, "AP75" : 100.00, "AP50" : 100.00}, ignore_index=True)
        df = df.append({"Model Name" : "System measures", "Iterations" : 5700, "mAP" : 95.578, "AP75" : 98.952, "AP50" : 98.970}, ignore_index=True)
        df = df.append({"Model Name" : "Stave measures", "Iterations" : 9000, "mAP" : 87.510, "AP75" : 96.744, "AP50" : 98.020}, ignore_index=True)
        # df = df.append({"Model Name" : "Combined", "Iterations" : 5700, "mAP" : 95.578, "AP75" : 98.952, "AP50" : 98.970}, ignore_index=True)
    elif model == "R_101_FPN_3x":
        df = df.append({"Model Name" : "Staves", "Iterations" : 15600, "mAP" : 94.293, "AP75" : 100.00, "AP50" : 100.00}, ignore_index=True)
        df = df.append({"Model Name" : "System measures", "Iterations" : 8700, "mAP" : 96.401, "AP75" : 98.864, "AP50" : 98.909}, ignore_index=True)
        df = df.append({"Model Name" : "Stave measures", "Iterations" : 6300, "mAP" : 87.476, "AP75" : 96.823, "AP50" : 98.020}, ignore_index=True)
        # df = df.append({"Model Name" : "Combined", "Iterations" : 5700, "mAP" : 95.578, "AP75" : 98.952, "AP50" : 98.970}, ignore_index=True)
    if type_of_annotation == "staves":
        st.table(df.loc[:0].set_index("Model Name"))
    elif type_of_annotation == "system_measures":
        st.table(df.loc[1:1].set_index("Model Name"))
    elif type_of_annotation == "stave_measures":
        st.table(df.loc[2:2].set_index("Model Name"))
    elif type_of_annotation == "system_measures-stave_measures-staves":
        st.write("THIS MODEL IS NOT READY YET!")
        return
        # st.table(df.loc[3:3].set_index("Model Name"))
    elif type_of_annotation == "model ensemble":
        st.table(df.set_index("Model Name"))
    
    if type_of_annotation == "model ensemble":
        for c in all_classes:
            st.markdown("# " + c)
            MetricsVisualiser().visualiseMetrics(root_dir, model, [c])
    else:
        metrics_type_annotations = [x for x in type_of_annotation.split("-") if x in all_classes]
        st.markdown("# " + type_of_annotation)
        MetricsVisualiser().visualiseMetrics(root_dir, model, metrics_type_annotations)

def handle_model_ensemble(model, display_original_image):
    img_file_buffer = st.file_uploader("Upload an image(s)", type=["png", "jpg", "jpeg"])
    if img_file_buffer == None:
        return
    for c in all_classes:
        model_dir = os.path.join(root_dir, "Models", model + "-" + c)
        cfg_file = "COCO-Detection/faster_rcnn_" + model + ".yaml"
        weight_file = "final_" + c + "_model.pth"
        path_to_weight_file = os.path.join(model_dir, weight_file)

        cfg = setup_cfg(1, cfg_file, path_to_weight_file)
        predictor = DefaultPredictor(cfg)

        if isinstance(img_file_buffer, list):
            for img_file in img_file_buffer:
                predict_image(predictor, img_file, display_original_image, custom_message=c)
        else:
            predict_image(predictor, img_file_buffer, display_original_image, custom_message=c)

def handle_standard_prediction(model, display_original_image, type_of_annotation):
    model_dir = os.path.join(root_dir, "Models", model + "-" + type_of_annotation)
    cfg_file = "COCO-Detection/faster_rcnn_" + model + ".yaml"
    weight_file = "final_" + type_of_annotation + "_model.pth"
    path_to_weight_file = os.path.join(model_dir, weight_file)
    display_multiple_classes = False

    which_classes = []
    if type_of_annotation == "system_measures-stave_measures-staves":
        cfg = setup_cfg(3, cfg_file, path_to_weight_file)
        display_multiple_classes = True
        which_classes = [all_classes.index(x) for x in type_of_annotation.split("-") if x in all_classes]
    else:
        cfg = setup_cfg(1, cfg_file, path_to_weight_file)
    predictor = DefaultPredictor(cfg)

    img_file_buffer = st.file_uploader("Upload an image(s)", type=["png", "jpg", "jpeg"])
    if img_file_buffer == None:
        return

    if isinstance(img_file_buffer, list):
        for img_file in img_file_buffer:
            predict_image(predictor, img_file, display_original_image, display_multiple_classes, which_classes)
    else:
        predict_image(predictor, img_file_buffer, display_original_image, display_multiple_classes, which_classes)

def predict_image(predictor, img_file, display_original_image, display_multiple_classes=False, which_classes=[], custom_message=""):
    image = Image.open(img_file).convert("RGB")
    if display_original_image:
        st.image(image, "Your uploaded image", use_column_width=True)

    im = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    outputs = predictor(im)
    # v = CustomVisualizer(im[:, :, ::-1], scale=1)

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

    # Set seed to negative to fully randomize everything.
    # Set seed to positive to use a fixed seed. Note that a fixed seed increases
    # reproducibility but does not guarantee fully deterministic behavior.
    # Disabling all parallelism further increases reproducibility.
    cfg.SEED = 1

    return cfg

if __name__ == "__main__":
    main()
