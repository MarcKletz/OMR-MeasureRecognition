import os
import json
import cv2
import urllib
import urllib.parse as urlparse
import urllib.request as urllib2
from tqdm import tqdm
import streamlit as st
from zipfile import ZipFile
import random

from detectron2.structures import BoxMode
from omrdatasettools.Downloader import Downloader
from omrdatasettools.OmrDataset import OmrDataset

base_github_url = "https://github.com/MarcKletz/OMR-MeasureRecognition/releases/download/"
release = "0.1"

# model_backbones = ["R_50_FPN_3x", "R_101_FPN_3x", "X_101_32x8d_FPN_3x"]
model_backbones = ["R_50_FPN_3x"]

annotations = ["system_measures", "stave_measures", "staves", "system_measures-staves", "system_measures-stave_measures-staves"]

class DataLoader:
    def download_datasets(self, root_dir):
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)

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

    def download_trained_models(self, root_dir):
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)

        file_path = os.path.join(root_dir, "Models")

        if not os.path.exists(file_path):
            os.mkdir(file_path)

        for backbone in model_backbones:
            if backbone == "R_50_FPN_3x":
                zip_file_size = 300
            elif backbone == "R_101_FPN_3x":
                zip_file_size = 450
            else:
                zip_file_size = 750

            for anno in annotations:
                path_to_folder = os.path.join(file_path, backbone + "-" + anno)
                if os.path.exists(path_to_folder):
                    print("folder path already exists: ", path_to_folder)
                    continue
                download_url = base_github_url + release + "/" + backbone + "-" + anno + ".zip"

                zip_path = path_to_folder + ".zip"
                self.__download_file(download_url, zip_path)

                with ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(file_path)
                os.remove(zip_path)

    def __download_file(self, url, dest):
        u = urllib2.urlopen(url)
        scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
        filename = os.path.basename(path)

        with open(dest, 'wb') as f:
            meta = u.info()
            meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
            meta_length = meta_func("Content-Length")
            file_size = None
            if meta_length:
                file_size = int(meta_length[0])

            if(st._is_running_with_streamlit):
                download_warning, progress_bar = None, None
                download_warning = st.warning("Downloading %s this might take a bit  ..." % filename)
                progress_bar = st.progress(0)
            else:
                print("Downloading: {0} \nBytes: {1} \nInto: {2}".format(filename, file_size, dest), flush=True)

            MEGABYTES = 2.0 ** 20.0
            file_size_dl = 0
            block_sz = 8192
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break

                file_size_dl += len(buffer)
                f.write(buffer)
                if file_size:
                    if(st._is_running_with_streamlit):
                        download_warning.warning("Downloading %s  ... (%6.2f/%6.2f MB)" % (filename, file_size_dl/MEGABYTES, file_size/MEGABYTES))
                        progress_bar.progress(min(file_size_dl/file_size, 1.0))
                    else:
                        print("\rDownloading %s... (%6.2f/%6.2f MB)" % (filename, file_size_dl/MEGABYTES, file_size/MEGABYTES), end="", flush=True)

    def get_CVC_Muscima_dicts(self, data_dir, image_dir, classes):
        idx = 0
        dataset_dicts = []

        for json_file in tqdm(sorted(os.listdir(data_dir))):
            with open(os.path.join(data_dir, json_file)) as f:
                imgs_anns = json.load(f)
        
            x = json_file.split("_")
            folder_name = x[1].lower()
            image_name = x[2].lower().replace("n-", "p0") + ".png"

            for augmentation_folder in sorted(os.listdir(image_dir)):
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

    def get_AudioLabs_dicts_from_json(self, data_dir, classes):
        idx = 0
        dataset_dicts = []

        for folder in tqdm(sorted(os.listdir(data_dir))):
            if folder == "coco":
                continue
            json_annotation_folder = os.path.join(data_dir, folder, "json")
            for json_file in sorted(os.listdir(json_annotation_folder)):
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

    def generateAllJsonDataAnnotations(self, root_dir):
        annotation_types = [["system_measures"], ["stave_measures"], ["staves"], ["system_measures", "staves"], ["system_measures", "stave_measures", "staves"]]

        muscima_data_dir = os.path.join(root_dir, "MuscimaPlusPlus_Measure_Annotations", "json")
        muscima_image_dir = os.path.join(root_dir, "CVC_Muscima_Augmented", "CVCMUSCIMA_MultiConditionAligned")

        audioLabs_data_dir = os.path.join(root_dir, "Measure_Bounding_Box_Annotations_v2")

        for annotation_type in annotation_types:
            json_pathname_extension = "-".join(str(elem) for elem in annotation_type)
            json_path = os.path.join(root_dir, "CVC_muscima_" + json_pathname_extension + ".json")
            if os.path.exists(json_path):
                print(json_path, " already exists!")
            else:
                muscima_data = self.get_CVC_Muscima_dicts(muscima_data_dir, muscima_image_dir, annotation_type)
                self.__save_to_json(json_path, muscima_data)

            json_path = os.path.join(root_dir, "AudioLabs_" + json_pathname_extension + ".json")
            if os.path.exists(json_path):
                print(json_path, "already exists!")
            else:
                audioLabs_data = self.get_AudioLabs_dicts_from_json(audioLabs_data_dir, annotation_type)
                self.__save_to_json(json_path, audioLabs_data)

    def load_from_json(self, json_path):
        with open(json_path) as json_file:
            return self.__convert_keys(json.load(json_file), self.__enum_to_names)

    def show_data(self, data, classes):
        print(data[0])
        print(classes)
        print(len(data))

    # This splits the data to 60% training, 20% testing and 20% validation
    def custom_muscima_split(self, muscima_data):
        rand = random.Random(1)

        muscima_split_data = []
        # get 140 pages
        for x in muscima_data:
            if "binary" in x["file_name"]:
                muscima_split_data.append(x)

        all_augmentations = ["binary", "grayscale", "interrupted", "kanungo", "staffline-thickness-variation-v1",
                        "staffline-thickness-variation-v2", "staffline-y-variation-v1", "staffline-y-variation-v2",
                        "typeset-emulation", "whitespeckles"]

        train_d = self.__split(all_augmentations, muscima_split_data, muscima_data, 0.6, rand)

        for x in train_d:
            if "binary" in x["file_name"]: # other folders are now in the split data aswel
                muscima_split_data.remove(x)

        test_d = self.__split(all_augmentations, muscima_split_data, muscima_data, 0.5, rand)

        for x in test_d:
            if "binary" in x["file_name"]: # other folders are now in the split data aswel
                muscima_split_data.remove(x)
        
        val_d = self.__split(all_augmentations, muscima_split_data, muscima_data, 1, rand)

        # just to make sure that the detectron data loader does not load the same image
        # back to back only with augmentations
        rand.shuffle(train_d)
        rand.shuffle(test_d)
        rand.shuffle(val_d)

        return train_d, test_d, val_d

    def __split(self, all_augmentations, muscima_split_data, muscima_data, percentage, random):
        data = []
        for d in random.sample(muscima_split_data, int(len(muscima_split_data) * percentage)):
            split_file_name = d["file_name"].replace("\\", "/").split("/")

            for augmentation in all_augmentations:
                augment_path = ""
                for i in range(0, len(split_file_name)):
                    if i == 5:
                        augment_path += augmentation
                    else:
                        augment_path += split_file_name[i]
                    if i != len(split_file_name)-1:
                        augment_path += "/"

                for msd in muscima_data:
                    if msd["file_name"].replace("\\", "/") == augment_path:
                        data.append(msd)
                        break
                
        return data

    # bbox_mode is from type BoxMode(IntEnum) which cant simply be converted to json
    # So I had to create functions that encode and decode the enums
    def __convert_keys(self, obj, convert):
        if isinstance(obj, list):
            return [self.__convert_keys(i, convert) for i in obj]
        if isinstance(obj, BoxMode):
            return convert(obj)
        if isinstance(obj, str) and obj.isupper(): # janky hack mate
            for x in BoxMode:
                if x.name == obj:
                    return self.__names_to_enum(obj)
        if not isinstance(obj, dict):
            return obj
        return {k: self.__convert_keys(v, convert) for k, v in obj.items()}
        
    def __enum_to_names(self, obj):
        return obj.name

    def __names_to_enum(self, obj):
        return BoxMode[obj]

    def __save_to_json(self, json_path, data):
        with open(json_path, "w") as outfile:
            json.dump(self.__convert_keys(data, self.__enum_to_names), outfile)
        print("saved to", outfile.name)

    def __get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None