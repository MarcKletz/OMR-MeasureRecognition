#%%
import os
from sklearn.model_selection import train_test_split

import detectron2
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer

from DataLoader import DataLoader

# %%
root_dir = "./../Data" # change this to download to a specific location on your pc
DataLoader().download_datasets(root_dir)
DataLoader().generateAllJsonDataAnnotations(root_dir)

# %%
# to decide which data should be loaded use this:

type_of_annotation = ["system_measures"]
# type_of_annotation = ["stave_measures"]
# type_of_annotation = ["staves"]

# type_of_annotation = ["system_measures", "staves"]
# type_of_annotation = ["system_measures", "stave_measures", "staves"]

json_pathname_extension = "-".join(str(elem) for elem in type_of_annotation)

# %%
json_path = os.path.join(root_dir, "CVC_muscima_" + json_pathname_extension + ".json")

muscima_data = DataLoader().load_from_json(json_path)
DataLoader().show_data(muscima_data, type_of_annotation)

# %%
json_path = os.path.join(root_dir, "AudioLabs_" + json_pathname_extension + ".json")

audioLabs_data = DataLoader().load_from_json(json_path)
DataLoader().show_data(audioLabs_data, type_of_annotation)

# %%
def registerDataset(data_name, d, data, classes):
    DatasetCatalog.register(data_name, lambda d=d: data)
    MetadataCatalog.get(data_name).set(thing_classes=classes)

    return MetadataCatalog.get(data_name)

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
def setup_cfg(train_data_name, test_data_name, num_classes, model_output_dir, cfg_file, existing_model_weight_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))

    cfg.OUTPUT_DIR = model_output_dir

    cfg.DATASETS.TRAIN = (train_data_name,)
    cfg.DATASETS.TEST = (test_data_name,)

    cfg.DATALOADER.NUM_WORKERS = 0 # Number of data loading threads

    cfg.MODEL.WEIGHTS = existing_model_weight_path

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # 128 faster, and good enough for toy dataset (default: 512)
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
network_type = "R_50_FPN_3x"
# network_type = "R_101_FPN_3x"

model_dir = os.path.join(root_dir, "Models", network_type + "-" + json_pathname_extension)

cfg_file = "COCO-Detection/faster_rcnn_" + network_type + ".yaml"

weight_file = os.path.join(model_dir, "last_checkpoint")
last_checkpoint = open(weight_file, "r").read()

path_to_weight_file = os.path.join(model_dir, last_checkpoint) 

cfg = setup_cfg(train_data_name, test_data_name, len(type_of_annotation), model_dir, cfg_file, path_to_weight_file)

#%%
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)

evaluator = COCOEvaluator(val_data_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, val_data_name)
print(inference_on_dataset(trainer.model, val_loader, evaluator))

# %%
