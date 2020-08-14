import io
from PIL import Image
import streamlit as st

# import some common libraries
import torch, torchvision
import numpy as np
import pandas as pd
import cv2
import os
import urllib
import requests

# import some detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor

from CustomVisualiser import CustomVisualizer

"""
# Inference with Detectron2 and a pretrained model
by Marc Kletz
"""

root_dir = "./../Data"
all_classes = ["system_measures", "stave_measures", "staves"]

# External files to download.
EXTERNAL_DEPENDENCIES = {
    os.path.join(root_dir, "Models", "R_50_FPN_3x-stave_measures", "final_stave_measures_model.pth"): {
        "id": "19znpzUKaLIa2_RZeukRpjOUCyz2Ynk8W",
        "size" : 315
    },
    os.path.join(root_dir, "Models", "R_50_FPN_3x-staves", "final_staves_model.pth"): {
        "id": "1_50H1mg2iaXxIXGlS976pp0dVxdXQJrS",
        "size" : 315
    },
    os.path.join(root_dir, "Models", "R_50_FPN_3x-system_measures", "final_system_measures_model.pth"): {
        "id": "1YiVuUXWnagXQcQpeCcz_n_qFAUqCqL0x",
        "size" : 315
    },
    os.path.join(root_dir, "Models", "R_50_FPN_3x-system_measures-stave_measures-staves", "final_system_measures-stave_measures-staves_model.pth"): {
        "id": "1Vo607A8x8GBZdos6YSwf9BwKKFoALODY",
        "size" : 315
    }
}

def main():
    # if the root dir does not exist
    # make all required directories
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
        model_dir = os.path.join(root_dir, "Models")
        os.mkdir(model_dir)

        for c in all_classes:
            trained_model_dir = os.path.join(model_dir, "R_50_FPN_3x-" + c)
            if not os.path.exists(trained_model_dir):
                os.mkdir(trained_model_dir)
        trained_model_dir = os.path.join(
            model_dir, "R_50_FPN_3x-system_measures-stave_measures-staves")
        os.mkdir(trained_model_dir)

    # download the models to these directories if they dont exist
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.title("What to do")

    model = st.sidebar.selectbox("Choose a model", ["R_50_FPN_3x", "R_101_FPN_3x"])
    type_of_annotation = st.sidebar.selectbox("Choose the type of annotation the model looks for",
        ["staves", "system_measures", "stave_measures", "system_measures-stave_measures-staves", "model ensemble"])
    display_original_image = st.sidebar.checkbox("Display the original image(s)", False)

    if type_of_annotation == "model ensemble":
        handle_model_ensemble(model, display_original_image)
    else:
        handle_standard_prediction(model, display_original_image, type_of_annotation)

def download_file(file_path):
    if not os.path.exists(file_path):
        download_warning, progress_bar = None, None
        try:
            download_warning = st.warning("Downloading %s this might take a bit..." % os.path.basename(file_path))
            progress_bar = st.progress(0)

            URL = "https://docs.google.com/uc?export=download"

            session = requests.Session()
            file_id = EXTERNAL_DEPENDENCIES[file_path]["id"]
            response = session.get(URL, params = { 'id' : file_id }, stream = True)
            token = get_confirm_token(response)

            if token:
                params = { 'id' : file_id, 'confirm' : token }
                response = session.get(URL, params = params, stream = True)

            counter = 0.0
            MEGABYTES = 2.0 ** 20.0
            CHUNK_SIZE = 32768
            length = EXTERNAL_DEPENDENCIES[file_path]["size"] * MEGABYTES

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        counter += len(chunk)
                        download_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" % (file_path, counter / MEGABYTES, length / MEGABYTES))
                        progress_bar.progress(min(counter / length, 1.0))
        finally:
            if download_warning is not None:
                download_warning.empty()
            if progress_bar is not None:
                progress_bar.empty()

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

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