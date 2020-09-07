import streamlit as st
import numpy as np
import PIL
import os, json, cv2, random
from IPython import display

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def main():
	st.write("HELLO WORLD")
	
	im = cv2.imread("./input.jpg")
	cv2_imshow(im)

    # the cv2_imshow function from google-colab package
def cv2_imshow(self, a):
	"""A replacement for cv2.imshow() for use in Jupyter notebooks.
	Args:
		a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
		(N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
		image.
	"""
	a = a.clip(0, 255).astype('uint8')
	# cv2 stores colors as BGR; convert to RGB
	if a.ndim == 3:
		if a.shape[2] == 4:
			a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
		else:
			a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
	display.display(PIL.Image.fromarray(a))

if __name__ == "__main__":
	main()