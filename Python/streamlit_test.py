import cv2
import streamlit as st

from ImageDisplayer import ImageDisplayer

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


"""
# Streamlit test
by Marc Kletz
"""


def main():
	st.write("Streamlit is working!")
	
	test_detectron()
	
def test_detectron():
	im = cv2.imread("../Images/detectron_test_image.jpg")
	ImageDisplayer().cv2_imshow(im)

if __name__ == "__main__":
	main()