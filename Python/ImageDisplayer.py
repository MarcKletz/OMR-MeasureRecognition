import cv2
import random
import PIL
from IPython import display
from CustomVisualizer import CustomVisualizer

class ImageDisplayer():

    def displayRandomSampleData(self, data, meta_data, page_count, category=None):
        for d in random.sample(data, page_count):
            print(d["file_name"])
            img = cv2.imread(d["file_name"])
            visualizer = CustomVisualizer(img[:, :, ::-1], metadata=meta_data, scale=1)
            vis = visualizer.draw_dataset_dict(d, category)
            self.cv2_imshow(vis.get_image()[:, :, ::-1])

    def displaySpecificSampleData(self, data, meta_data, path_to_page, category=None):
        d = [x for x in data if x["file_name"] == path_to_page][0]
        print(d["file_name"])
        img = cv2.imread(d["file_name"])
        visualizer = CustomVisualizer(img[:, :, ::-1], metadata=meta_data, scale=1)
        vis = visualizer.draw_dataset_dict(d, category)
        self.cv2_imshow(vis.get_image()[:, :, ::-1])

    def displayRandomPredictData(self, predictor, data, meta_data, sample_ammount=3, category=None):
        for d in random.sample(data, sample_ammount):    
            print(d["file_name"])
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)
            v = CustomVisualizer(im[:, :, ::-1], metadata=meta_data, scale=1)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"), category)
            self.cv2_imshow(v.get_image()[:, :, ::-1])

    def displaySpecificPredictData(self, predictor, path_to_page, category=None): 
        print(path_to_page)
        im = cv2.imread(path_to_page)
        outputs = predictor(im)
        v = CustomVisualizer(im[:, :, ::-1], scale=1)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"), category)
        self.cv2_imshow(v.get_image()[:, :, ::-1])

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