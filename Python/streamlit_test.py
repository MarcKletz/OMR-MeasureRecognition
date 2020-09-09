from PIL import Image
import streamlit as st

from CustomVisualizer import CustomVisualizer

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
	image = Image.open("./Images/detectron_test_image.jpg").convert("RGB")
	st.image(image, "Your uploaded image", use_column_width=True)
	
	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
	predictor = DefaultPredictor(cfg)
	outputs = predictor(im)
	
	v = CustomVisualizer(im[:, :, ::-1], scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    st.image(v.get_image()[:, :, ::-1], "Network prediction", use_column_width=True)

if __name__ == "__main__":
	main()