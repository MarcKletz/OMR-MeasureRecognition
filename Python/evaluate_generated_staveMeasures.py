#%%
import os, sys
import pandas as pd
import numpy as np
from tqdm import tqdm

from DataLoader import DataLoader
from ImageDisplayer import ImageDisplayer

#
# This evaluation script is a result of copying and modifying the existing evaluation code from:
# https://github.com/apacha/MusicObjectDetection/blob/master/MusicObjectDetection/evaluation/evaluate.py
#

#%%
root_dir = "./../Data" # change this to download to a specific location on your pc

network_type = "R_50"
# network_type = "R_101"
# network_type = "X_101"

network_used = "SingleNetwork"
# network_used = "TwoNN_SystemAndStaves"

data_frame = pd.read_csv(os.path.join(root_dir, network_type + "_" + network_used + "_StaveMeasures.csv"))
data_frame.head()
# %%
type_of_annotation = ["stave_measures"]
json_pathname_extension = "-".join(str(elem) for elem in type_of_annotation)

json_path = os.path.join(root_dir, "CVC_muscima_" + json_pathname_extension + ".json")
muscima_data = DataLoader().load_from_json(json_path)

json_path = os.path.join(root_dir, "AudioLabs_" + json_pathname_extension + ".json")
audioLabs_data = DataLoader().load_from_json(json_path)

# %%
from sklearn.model_selection import train_test_split

# Put all pages for an augmentation into one set (training, test, validation)
# this code makes sure that a music page will be in one set with all their augmentations
# we do not want the same music pages with different augmentations ending up in the training and test dataset. (data-leak)
musicma_train_data, musicma_test_data, musicma_val_data = DataLoader().custom_muscima_split(muscima_data)

audiolabs_train_data, test_val_data = train_test_split(audioLabs_data, test_size=0.4, random_state=1)
audiolabs_test_data, audiolabs_val_data = train_test_split(test_val_data, test_size=0.5, random_state=1)

test_data = musicma_test_data + audiolabs_test_data

# %%
# the path slashes are different and need to be compared later, normalise them
for index, item in enumerate(test_data):
    test_data[index]["file_name"] = item["file_name"].replace("\\", "/")

for index, row in data_frame.iterrows():
    data_frame["Image"][index] = row["Image"].replace("\\", "/")

images = [i["file_name"] for i in test_data]

# %%
ann_boxes = []
for image in test_data:
    for annotation in image["annotations"]:
        box = {
            "image" : image["file_name"],
            "left" : float(annotation["bbox"][0]),
            "top" : float(annotation["bbox"][1]),
            "right" : float(annotation["bbox"][2]),
            "bottom" : float(annotation["bbox"][3]),
            "score" : None
        }
        ann_boxes.append(box)

# %%
pred_boxes = []
for index, row in data_frame.iterrows():
    box = {
        "image" : row["Image"],
        "left" : row["Left"],
        "top" : row["Top"],
        "right" : row["Right"],
        "bottom" : row["Bottom"],
        "score" : None
    }
    pred_boxes.append(box)

# %%
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def compute_overlap(boxes, query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N,K), dtype=np.float64)
    
    for k in range(K):
        box_area = ((query_boxes[k][2] - query_boxes[k][0] + 1) * (query_boxes[k][3] - query_boxes[k][1] + 1))
        
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - (iw * ih)
                    )
                    
                    overlaps[n, k] = iw * ih / ua
                    
    return overlaps

def get_metrics(average_precisions):
    meanAP = 0
    count = 0
    
    for category in average_precisions:
        meanAP += average_precisions[0]
        count += 1
        
    meanAP /= count
    
    return meanAP

def get_ap(images, pred_boxes, ann_boxes, iou_threshold):
    average_precisions = []

    false_positives = np.zeros((0,))
    true_positives = np.zeros((0,))
    num_annotations = 0

    for image in images:
        image_annotations = [box for box in ann_boxes if box["image"] == image]
        image_annotations_boxes = np.array([[b["left"], b["top"], b["right"], b["bottom"]] for b in image_annotations])
        num_annotations += len(image_annotations)

        image_predictions = [box for box in pred_boxes if box["image"] == image]
        
        detected_annotations = []

        for prediction in image_predictions:
            if len(image_annotations) == 0:
                false_positives = np.append(false_positives, 1)
                true_positives = np.append(true_positives, 0)
                continue

            prediction_box = np.array([prediction["left"], prediction["top"], prediction["right"], prediction["bottom"]])
            overlaps = compute_overlap(np.expand_dims(prediction_box, axis=0), image_annotations_boxes)

            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_annotation]

            if max_overlap > iou_threshold and assigned_annotation not in detected_annotations:
                false_positives = np.append(false_positives, 0)
                true_positives = np.append(true_positives, 1)
                detected_annotations.append(assigned_annotation)
            else:
                false_positives = np.append(false_positives, 1)
                true_positives = np.append(true_positives, 0)

    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives = np.cumsum(true_positives)

    # compue recall and precision
    recall = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
    
    # compute average precision
    average_precision = compute_ap(recall, precision)
    average_precisions = [average_precision, num_annotations]

    return average_precisions


# %%
thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
ap_by_th = {}

for th in tqdm(thresholds):  
    ap_by_th[th] = get_ap(images, pred_boxes, ann_boxes, iou_threshold = th)

# %%
global_ap = [0, 0]

for th in ap_by_th:
    ap, count = ap_by_th[th]
    global_ap[0] += ap
    global_ap[1] += count
    
global_ap[0] /= len(ap_by_th)
global_ap[1] /= len(ap_by_th)

mAP = get_metrics(global_ap)
print('COCO, IoU = [0.5:0.95]')
print('mAP', mAP, '\nAP75', ap_by_th[0.75][0], '\nAP50', ap_by_th[0.5][0])

# %%

