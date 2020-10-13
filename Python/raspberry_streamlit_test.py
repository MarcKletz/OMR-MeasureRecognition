import ctypes
libgcc_s = ctypes.CDLL("/usr/lib/libgcc_s.so.1")

import cv2

import torch, torchvision

import io
import numpy as np
import pandas as pd
import os

import numpy as np
from PIL import Image
import streamlit as st

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
# Streamlit test
by Marc Kletz
"""


def main():
    st.write("Streamlit is working!")

    test_detectron()

def test_detectron():
    st.write("in test detectron")	

    image = Image.open("./Images/detectron_test_image.jpg").convert("RGB")
    st.image(image, "Your uploaded image", use_column_width=True)
    im = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    if not torch.cuda.is_available():
        st.write("NO NVIDIA GPU FOUND - FALLBACK TO CPU")
        cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    v = CustomVisualizer(im[:, :, ::-1], scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    st.image(v.get_image()[:, :, ::-1], "Network prediction", use_column_width=True)

if __name__ == "__main__":
    main()
