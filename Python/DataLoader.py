import os
import json
import cv2
from tqdm import tqdm

from detectron2.structures import BoxMode

from omrdatasettools.Downloader import Downloader
from omrdatasettools.OmrDataset import OmrDataset

class DataLoader:
    def download_datasets(self, root_dir):
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

    def get_CVC_Muscima_dicts(self, data_dir, image_dir, classes):
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

    def get_AudioLabs_dicts_from_json(self, data_dir, classes):
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

    def generateAllJsonDataAnnotations(self, root_dir):
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