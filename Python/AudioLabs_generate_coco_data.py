# %%
import json
import os
import pandas as pd
import cv2

# %% GENERATE COCO DATA
json_dict = {}

# %%
info = {}
info["description"] = "AudioLabs v2.0 dataset for Measure Detection"
info["url"] = "https://apacha.github.io/OMR-Datasets/"
info["version"] = "2.2"
info["year"] = "2020"
info["contributor"] = "Marc Kletz, Frank Zalkow, Angel Villar Corrales, TJ Tsai, Vlora Arifi-Müller, and Meinard Müller"
info["date_created"] = "2020/07/30"

# %%
licenses = []

license_1 = {}
license_1["id"] = "1"
license_1["name"] = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)"

license_2 = {}
license_2["id"] = "2"
license_2["name"] = "Public Domain (source: IMSLP)"

license_3 = {}
license_3["id"] = "3"
license_3["name"] = "Provided by kind permission of Carus publishing house"

licenses.append(license_1)
licenses.append(license_2)
licenses.append(license_3)

# %%
categories = []

categorie_1 = {}
categorie_1["id"] = "1"
categorie_1["name"] = "system_measure"
categorie_1["supercategory"] = "region"

categorie_2 = {}
categorie_2["id"] = "2"
categorie_2["name"] = "stave_measure"
categorie_2["supercategory"] = "region"

categorie_3 = {}
categorie_3["id"] = "3"
categorie_3["name"] = "stave"
categorie_3["supercategory"] = "region"

categories.append(categorie_1)
categories.append(categorie_2)
categories.append(categorie_3)

# %%
path_to_audiolabs_folder = "./../Data/Measure_Bounding_Box_Annotations_v2"

images = []
idx = 1

for folder in os.listdir(path_to_audiolabs_folder):
    if folder == "coco":
        continue
    path_to_img_folder = os.path.join(path_to_audiolabs_folder, folder, "img")
    if os.path.exists(path_to_img_folder):
        for img in os.listdir(path_to_img_folder):
            height, width = cv2.imread(os.path.join(path_to_img_folder, img)).shape[:2]

            if "Beethoven" in img or "Shubert" in img or "Wagner" in img:
                license = 2
            elif "Chorissimo" in img:
                license = 3

            img_annotation = {}
            img_annotation["id"] = idx
            img_annotation["file_name"] = img
            img_annotation["width"] = width
            img_annotation["height"] = height
            img_annotation["date_captured"] = ""
            img_annotation["license"] = license
            img_annotation["coco_url"] = ""
            img_annotation["flickr_url"] = ""

            images.append(img_annotation)
            idx += 1

# %%
def generate_coco_annotations(annotations, df, idx, img_id, category_id):
    for index, row in df.iterrows():
        anno = {}
        anno["id"] = idx
        anno["image_id"] = img_id
        anno["category_id"] = category_id
        anno["iscrowd"] = 0
        anno["area"] = 0.0
        anno["bbox"] = [row["Left"], row["Top"], row["Width"], row["Height"]]
        anno["segmentation"] = []
        
        annotations.append(anno)

        idx += 1

    return idx

#%%
annotations = []
idx = 1
img_id = 1

for folder in os.listdir(path_to_audiolabs_folder):
    if folder == "coco":
        continue

    annotation_folder = os.path.join(path_to_audiolabs_folder, folder, "csv")
    all_files = os.listdir(annotation_folder)
    all_files.sort()

    # to get the correct ordering of annotations as system_measure, stave_measure, stave
    system_measures = all_files[2]
    stave_measures = all_files[0]
    staves = all_files[1]

    system_df = pd.read_csv(os.path.join(annotation_folder, system_measures))
    measure_df = pd.read_csv(os.path.join(annotation_folder, stave_measures))
    stave_df = pd.read_csv(os.path.join(annotation_folder, staves))

    image_names = []
    for imgs in os.listdir(os.path.join(path_to_audiolabs_folder, folder, "img")):
        image_names.append(imgs)

    for img_name in image_names:
        if img_name == "Schubert_D911-01_000.png":
            print("?")
        system_page = system_df[system_df["Image"] == img_name]
        measure_page = measure_df[measure_df["Image"] == img_name]
        stave_page = stave_df[stave_df["Image"] == img_name]

        idx = generate_coco_annotations(annotations, system_page, idx, img_id, 1)
        idx = generate_coco_annotations(annotations, measure_page, idx, img_id, 2)
        idx = generate_coco_annotations(annotations, stave_page, idx, img_id, 3)

        img_id += 1

# %%
json_dict["info"] = info
json_dict["licenses"] = licenses
json_dict["categories"] = categories
json_dict["images"] = images
json_dict["annotations"] = annotations

# %%
path_to_coco_dir = os.path.join(path_to_audiolabs_folder, "coco")
if not os.path.exists(path_to_coco_dir):
    os.mkdir(path_to_coco_dir)

# %%
with open(os.path.join(path_to_audiolabs_folder, "coco", "all_annotations.json"), "w", encoding="utf8") as outfile:
    json.dump(json_dict, outfile, indent=4, ensure_ascii=False)

# %%
# -----------------------------

# %% GENERATE JSON DATA
def generate_json_annotations(df):
    annotation = []
    for index, row in df.iterrows():
        anno = {}
        anno["left"] = row["Left"]
        anno["top"] = row["Top"]
        anno["height"] = row["Height"]
        anno["width"] = row["Width"]
        annotation.append(anno)

    return annotation
# %%
path_to_audiolabs_folder = "./../Data/Measure_Bounding_Box_Annotations_v2"

for folder in os.listdir(path_to_audiolabs_folder):
    if folder == "coco":
        continue

    annotation_folder = os.path.join(path_to_audiolabs_folder, folder, "csv")
    all_files = os.listdir(annotation_folder)
    all_files.sort()

    # to get the correct ordering of annotations as system_measure, stave_measure, stave
    system_measures = all_files[2]
    stave_measures = all_files[0]
    staves = all_files[1]

    system_df = pd.read_csv(os.path.join(annotation_folder, system_measures))
    measure_df = pd.read_csv(os.path.join(annotation_folder, stave_measures))
    stave_df = pd.read_csv(os.path.join(annotation_folder, staves))

    image_names = []
    for imgs in os.listdir(os.path.join(path_to_audiolabs_folder, folder, "img")):
        image_names.append(imgs)

    for img_name in image_names:
        annotations = {}
        height, width = cv2.imread(os.path.join(path_to_audiolabs_folder, folder, "img", img_name)).shape[:2]

        system_page = system_df[system_df["Image"] == img_name]
        measure_page = measure_df[measure_df["Image"] == img_name]
        stave_page = stave_df[stave_df["Image"] == img_name]

        annotations["width"] = width
        annotations["height"] = height
        annotations["system_measures"] = generate_json_annotations(system_page)
        annotations["stave_measures"] = generate_json_annotations(measure_page)
        annotations["staves"] = generate_json_annotations(stave_page)
        
        json_folder = os.path.join(path_to_audiolabs_folder, folder, "json")
        if not os.path.exists(json_folder):
            os.mkdir(json_folder)

        with open(os.path.join(json_folder, img_name.split(".")[0] + ".json"), "w", encoding="utf8") as outfile:
            json.dump(annotations, outfile, indent=4, ensure_ascii=False)


# %%
