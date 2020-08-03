#%%
import torch, torchvision

# import some common libraries
import numpy as np
import pandas as pd
import cv2
import os
import random
import json
from tqdm import tqdm
from matplotlib import pyplot as plt

# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.structures.boxes import Boxes
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# %%
from omrdatasettools.Downloader import Downloader
from omrdatasettools.OmrDataset import OmrDataset

def download_datasets(root_dir):
    muscima_plus_plus_path = os.path.join(root_dir, "MuscimaPlusPlus_V2")
    if not os.path.exists(muscima_plus_plus_path):
        Downloader().download_and_extract_dataset(OmrDataset.MuscimaPlusPlus_V2, muscima_plus_plus_path)
    else:
        print("MuscimaPlusPlus_V2 already exists")

    cvc_muscima_path = os.path.join(root_dir, "CVC_Muscima_Augmented")
    if not os.path.exists(cvc_muscima_path):
        Downloader().download_and_extract_dataset(OmrDataset.CvcMuscima_MultiConditionAligned, cvc_muscima_path)
    else:
        print("CVC_Muscima_Augmented already exists")

    muscima_plus_plus_measure_annotations_path = os.path.join(root_dir, "MuscimaPlusPlus_Measure_Annotations")
    if not os.path.exists(muscima_plus_plus_measure_annotations_path):
        Downloader().download_and_extract_dataset(OmrDataset.MuscimaPlusPlus_MeasureAnnotations, muscima_plus_plus_measure_annotations_path)
    else:
        print("MuscimaPlusPlus_Measure_Annotations already exists")

    measure_bounding_box_annotations_v2_path = os.path.join(root_dir)
    if not os.path.exists(measure_bounding_box_annotations_v2_path):
        Downloader().download_and_extract_dataset(OmrDataset.MeasureBoundingBoxAnnotations_v2, measure_bounding_box_annotations_v2_path)
    else:
        print("Measure_Bounding_Box_Annotations_v2 already exists")

root_dir = "./../Data" # change this to download to a specific location on your pc
download_datasets(root_dir)

# %%
# bbox_mode is from type BoxMode(IntEnum) which cant simply be converted to json
# So I had to create functions that encode and decode the enums
def convert_keys(obj, convert):
    if isinstance(obj, list):
        return [convert_keys(i, convert) for i in obj]
    if isinstance(obj, BoxMode):
        return convert(obj)
    if isinstance(obj, str) and obj.isupper(): # janky hack mate
        for x in BoxMode:
            if x.name == obj:
                return names_to_enum(obj)
    if not isinstance(obj, dict):
        return obj
    return {k: convert_keys(v, convert) for k, v in obj.items()}

def enum_to_names(obj):
    return obj.name

def names_to_enum(obj):
    return BoxMode[obj]

def save_to_json(json_path, data):
    with open(json_path, "w") as outfile:
        json.dump(convert_keys(data, enum_to_names), outfile)
    print("saved to", outfile.name)

def load_from_json(json_path):
    with open(json_path) as json_file:
        return convert_keys(json.load(json_file), names_to_enum)

def show_data(data, classes):
    print(data[0])
    print(classes)
    print(len(data))

# %%
def get_CVC_Muscima_dicts(data_dir, image_dir, classes):
    idx = 0
    dataset_dicts = []

    for json_file in tqdm(os.listdir(data_dir)):
        with open(os.path.join(data_dir, json_file)) as f:
            imgs_anns = json.load(f)
    
        x = json_file.split("_")
        folder_name = x[1].lower()
        image_name = x[2].lower().replace("n-", "p0") + ".png"

        for augmentation_folder in os.listdir(image_dir):
            path_to_image_file = os.path.join(image_dir, augmentation_folder, folder_name, image_name)
            
            record = {}
            record["file_name"] = path_to_image_file
            record["image_id"] = idx
            idx += 1
            record["height"] = imgs_anns["height"]
            record["width"] = imgs_anns["width"]

            objs = []
            for c in classes:
                for anno in imgs_anns[c]:
                    obj = {
                        "bbox" : [anno["left"], anno["top"], anno["right"], anno["bottom"]],
                        "bbox_mode" : BoxMode.XYXY_ABS,
                        "category_id" : classes.index(c),
                        "iscrowd" : 0
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    
    return dataset_dicts
# %%
def get_AudioLabs_dicts_from_json(data_dir, classes):
    idx = 0
    dataset_dicts = []

    for folder in tqdm(os.listdir(data_dir)):
        if folder == "coco":
            continue
        json_annotation_folder = os.path.join(data_dir, folder, "json")
        for json_file in os.listdir(json_annotation_folder):
            with open(os.path.join(json_annotation_folder, json_file)) as f:
                imgs_anns = json.load(f)

            image_name = os.path.splitext(json_file)[0]
            path_to_image_file = os.path.join(data_dir, folder, "img", image_name + ".png")
            height, width = cv2.imread(path_to_image_file).shape[:2]

            record = {}
            record["file_name"] = path_to_image_file
            record["image_id"] = idx
            idx += 1
            record["height"] = height
            record["width"] = width
                    
            objs = []
            for c in classes:
                for anno in imgs_anns[c]:
                    left = anno["left"]
                    top = anno["top"]
                    width = anno["width"]
                    height = anno["height"]

                    if type(left) == str or type(width) == str or type(top) == str or type(height) == str:
                        print("?")

                    right, bottom = (left + width), (top + height)
                    
                    obj = {
                        "bbox" : [left, top, right, bottom],
                        "bbox_mode" : BoxMode.XYXY_ABS,
                        "category_id" : classes.index(c),
                        "iscrowd" : 0
                    }
                    objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts

# %%
def generateAllJsonDataAnnotations():
    annotation_types = [["system_measures"], ["stave_measures"], ["staves"], ["system_measures", "stave_measures", "staves"]]

    muscima_data_dir = os.path.join(root_dir, "MuscimaPlusPlus_Measure_Annotations", "json")
    muscima_image_dir = os.path.join(root_dir, "CVC_Muscima_Augmented", "CVCMUSCIMA_MultiConditionAligned")

    audioLabs_data_dir = os.path.join(root_dir, "Measure_Bounding_Box_Annotations_v2")

    for annotation_type in annotation_types:
        json_pathname_extension = "-".join(str(elem) for elem in annotation_type)
        json_path = os.path.join(root_dir, "CVC_muscima_" + json_pathname_extension + ".json")
        if os.path.exists(json_path):
            print(json_path, " already exists!")
        else:
            muscima_data = get_CVC_Muscima_dicts(muscima_data_dir, muscima_image_dir, annotation_type)
            save_to_json(json_path, muscima_data)

        json_path = os.path.join(root_dir, "AudioLabs_" + json_pathname_extension + ".json")
        if os.path.exists(json_path):
            print(json_path, " already exists!")
        else:
            audioLabs_data = get_AudioLabs_dicts_from_json(audioLabs_data_dir, annotation_type)
            save_to_json(json_path, audioLabs_data)
# %%
generateAllJsonDataAnnotations()

#%%
# to decide which data should be loaded use this:

# type_of_annotation = ["system_measures"]
# type_of_annotation = ["stave_measures"]
type_of_annotation = ["staves"]

# type_of_annotation = ["system_measures", "stave_measures", "staves"]

json_pathname_extension = "-".join(str(elem) for elem in type_of_annotation)

#%%
json_path = os.path.join(root_dir, "CVC_muscima_" + json_pathname_extension + ".json")

muscima_data = load_from_json(json_path)
show_data(muscima_data, type_of_annotation)

#%%
json_path = os.path.join(root_dir, "AudioLabs_" + json_pathname_extension + ".json")

audioLabs_data = load_from_json(json_path)
show_data(audioLabs_data, type_of_annotation)

# %%
def registerDataset(data_name, d, data, classes):
    DatasetCatalog.register(data_name, lambda d=d: data)
    MetadataCatalog.get(data_name).set(thing_classes=classes)

    return MetadataCatalog.get(data_name)

# %%
class MyVisualizer(Visualizer):
    def _create_text_labels(self, classes, scores, class_names):
        """
        Args:
            classes (list[int] or None):
            scores (list[float] or None):
            class_names (list[str] or None):
        Returns:
            list[str] or None
        """
        labels = None
        if classes is not None and class_names is not None and len(class_names) > 1:
            labels = [class_names[i] for i in classes]
        if scores is not None:
            if labels is None:
                labels = ["{:.0f}%".format(s * 100) for s in scores]
            else:
                labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
        return labels

    def draw_dataset_dict(self, dic, category=None):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.

        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.
            category: the integer category for the desired annotation to display as a list or None if all of them

        Returns:
            output (VisImage): image object with visualizations.
        """
        # start additional code
        unfiltered_annos = dic.get("annotations", None)
        if category == None:
            annos = unfiltered_annos
        else:
            annos = [] 
            for annotations in unfiltered_annos:
                if annotations["category_id"] in category:
                    annos.append(annotations)
        # end additional code

        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS) for x in annos]

            labels = [x["category_id"] for x in annos]
            colors = None
            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                colors = [
                    self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in labels
                ]
            names = self.metadata.get("thing_classes", None)
            if names:
                labels = [names[i] for i in labels]
            labels = [
                "{}".format(i) + ("|crowd" if a.get("iscrowd", 0) else "")
                for i, a in zip(labels, annos)
            ]
            self.overlay_instances(
                labels=labels, 
                boxes=boxes, 
                masks=masks, 
                keypoints=keypts, 
                assigned_colors=colors,
                alpha=1.0 # added alpha to be 1.0
            )

        sem_seg = dic.get("sem_seg", None)
        if sem_seg is None and "sem_seg_file_name" in dic:
            with PathManager.open(dic["sem_seg_file_name"], "rb") as f:
                sem_seg = Image.open(f)
                sem_seg = np.asarray(sem_seg, dtype="uint8")
        if sem_seg is not None:
            self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)
        return self.output

    # might not work for every useage - but works for me
    # have not tested with keypoints and masks since my model does not have these
    def draw_instance_predictions(self, predictions, category=None):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
            category: the integer category for the desired annotation to display as a list or None if all of them

        Returns:
            output (VisImage): image object with visualizations.
        """

        # start additional code
        if category == None:
            boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
            scores = predictions.scores if predictions.has("scores") else None
            classes = predictions.pred_classes if predictions.has("pred_classes") else None
            labels = self._create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
            keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        else:
            all_boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
            all_scores = predictions.scores if predictions.has("scores") else None
            all_classes = predictions.pred_classes if predictions.has("pred_classes") else None
            all_labels = self._create_text_labels(all_classes, all_scores, self.metadata.get("thing_classes", None))
            all_keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

            boxes = [] if all_boxes != None else None
            scores = [] if all_scores != None else None
            classes = [] if all_classes != None else None
            labels = [] if all_labels != None else None
            keypoints = [] if all_keypoints != None else None

            for c in category:
                for i in range(0, len(all_classes)):
                    print(all_classes[i], c)
                    if all_classes[i] == c:
                        classes.append(all_classes[i])

                        if all_boxes != None:
                            boxes.append(all_boxes[i])
                        if all_scores != None:
                            scores.append(all_scores[i])
                        if all_labels != None:
                            labels.append(all_labels[i])
                        if all_keypoints != None:
                            keypoints.append(all_keypoints[i])

            if boxes != None and len(boxes) > 0:
                boxes = Boxes(torch.cat([b.tensor for b in boxes], dim=0))
            if scores != None and len(scores) > 0:
                scores = torch.stack(scores)
            if classes != None and len(classes) > 0:
                classes = torch.stack(classes)
        # end additional code

        # removed alpha from here and put it as fixed value
        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None
        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
        else:
            colors = None
        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
            )

        self.overlay_instances(
            labels=labels,
            boxes=boxes,
            masks=masks,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=1.0, # changed alpha to be 1.0
        )
        return self.output

# %% the cv2_imshow from google-colab package
from IPython import display
import PIL

def cv2_imshow(a):
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

#%%
def displayRandomSampleData(data, meta_data, page_count, category=None):
    for d in random.sample(data, page_count):
        print(d["file_name"])
        img = cv2.imread(d["file_name"])
        visualizer = MyVisualizer(img[:, :, ::-1], metadata=meta_data, scale=1)
        vis = visualizer.draw_dataset_dict(d, category)
        cv2_imshow(vis.get_image()[:, :, ::-1])

def displaySpecificSampleData(data, meta_data, path_to_page, category=None):
    d = [x for x in data if x["file_name"] == path_to_page][0]
    print(d["file_name"])
    img = cv2.imread(d["file_name"])
    visualizer = MyVisualizer(img[:, :, ::-1], metadata=meta_data, scale=1)
    vis = visualizer.draw_dataset_dict(d, category)
    cv2_imshow(vis.get_image()[:, :, ::-1])

def displayRandomPredictData(data, meta_data, sample_ammount=3, category=None):
    for d in random.sample(data, sample_ammount):    
        print(d["file_name"])
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = MyVisualizer(im[:, :, ::-1], metadata=meta_data, scale=1)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"), category)
        cv2_imshow(v.get_image()[:, :, ::-1])

def displaySpecificPredictData(data, meta_data, path_to_page, category=None): 
    print(path_to_page)
    im = cv2.imread(path_to_page)
    outputs = predictor(im)
    v = MyVisualizer(im[:, :, ::-1], metadata=meta_data, scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"), category)
    cv2_imshow(v.get_image()[:, :, ::-1])

# %%
from sklearn.model_selection import train_test_split

musicma_train_data, test_val_data = train_test_split(muscima_data, test_size=0.4, random_state=1)
musicma_test_data, musicma_val_data = train_test_split(test_val_data, test_size=0.5, random_state=1)

audiolabs_train_data, test_val_data = train_test_split(audioLabs_data, test_size=0.4, random_state=1)
audiolabs_test_data, audiolabs_val_data = train_test_split(test_val_data, test_size=0.5, random_state=1)

train_data = musicma_train_data + audiolabs_train_data
test_data = musicma_test_data + audiolabs_test_data
val_data = musicma_val_data + audiolabs_val_data

train_data_name = "train"
metadata = registerDataset(train_data_name, train_data_name, train_data, type_of_annotation)

test_data_name = "test"
registerDataset(test_data_name, test_data_name, test_data, type_of_annotation)

val_data_name = "val"
registerDataset(val_data_name, val_data_name, val_data, type_of_annotation)

# %%
displayRandomSampleData(train_data, metadata, 3, [type_of_annotation.index("staves")])

#%%
def setup_cfg(num_classes, cfg_file, existing_model_weight_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))

    cfg.MODEL.WEIGHTS = existing_model_weight_path

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set the testing threshold for this model. Model should be at least 20% confident detection is correct
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.2

    # Set seed to negative to fully randomize everything.
    # Set seed to positive to use a fixed seed. Note that a fixed seed increases
    # reproducibility but does not guarantee fully deterministic behavior.
    # Disabling all parallelism further increases reproducibility.
    cfg.SEED = 1

    return cfg

# %%
model_dir = os.path.join(root_dir, "Models", "R_101_FPN_3x-staves")

cfg_file = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml" # faster training, but slightly less AP, way smaller model (.pth) file
weight_file = "final_staves_model.pth"
path_to_weight_file = os.path.join(model_dir, weight_file) 

cfg = setup_cfg(len(type_of_annotation), cfg_file, path_to_weight_file)

# %%
predictor = DefaultPredictor(cfg)

# %%
displayRandomPredictData(test_data, metadata, 5)

# %%
