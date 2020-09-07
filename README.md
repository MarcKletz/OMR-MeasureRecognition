# THIS REPO AND README IS STILL UNDER HEAVY DEVELOPMENT!

# About this repo

All data required for this github project can be downloaded by using the provided DaterLoader script. <br>
Simply calling: <br>
```
DataLoader().download_datasets(root_dir)
DataLoader().download_trained_models(root_dir)
DataLoader().generateAllJsonDataAnnotations(root_dir)
``` 
Will download the datasets from the OMR-Datasets repository, <br>
download all the trained models as specified in the overview and <br>
generate the json annotations for the datasets.

## Faster R-CNN and ResNet-50 backbone
|   Model Name        |   Iterations  |    mAP   |    AP75  |    AP50  |
|:-------------------:|:-------------:|:--------:|:--------:|:--------:|
|   System measures   |     5700      |  95.578  |  98.952  |  98.970  |
|   Stave measures    |     9000      |  87.510  |  96.744  |  98.020  |
|      Staves         |     5700      |  93.173  |  100.00  |  100.00  |
|      Combined       |     TODO      |  TODO  |  TODO  |  TODO  |

## scores for the combined dataset per category
| Category        | mAP    | Category       | mAP    | Category   | mAP    |
|:---------------:|:------:|:--------------:|:------:|:----------:|:------:|
| system_measures | TODO | stave_measures | TODO | staves     | TODO |

## Faster R-CNN and ResNet-101 backbone
|   Model Name        |   Iterations  |    mAP   |    AP75  |    AP50  |
|:-------------------:|:-------------:|:--------:|:--------:|:--------:|
|   System measures   |     8700      |  96.401  |  98.864  |  98.909  |
|   Stave measures    |     6300      |  87.476  |  96.823  |  98.020  |
|      Staves         |     15600     |  94.293  |  100.00  |  100.00  |
|      Combined       |     TODO      |  TODO  |  TODO  |  TODO  |

## scores for the combined dataset per category
| Category        | mAP    | Category       | mAP    | Category   | mAP    |
|:---------------:|:------:|:--------------:|:------:|:----------:|:------:|
| system_measures | TODO | stave_measures | TODO | staves     | TODO |

# Intallation Setup

Requirements before starting: <br>
Python >= 3.6 <br>
Git <br>
Cuda Toolkit 10.1 <br>

## For Linux:

install all the required python librarys <br>
pip install -r linux_requirements.txt

## For Windows:

Prereqs: <br>
Windows SDK <br>
C++14 build tools <br>

Step 1: <br>
install all the required python librarys <br>
from OMR-
`pip install -r Python/windows_requirements.txt` <br>

Step 2: <br>
manually install detectron2 because there is no windows support, we need to pull and install from source <br>
```
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
git reset --hard be792b959bca9af0aacfa04799537856c7a92802 # to pull detectron version 0.2.1
```
change the following line in detectron2\detectron2\layers\csrc\cocoeval\cocoeval.cpp(483): <br>
localtime_r(&rawtime, &local_time) to localtime_s(&local_time, &rawtime); <br>
solution from : https://github.com/conansherry/detectron2/issues/2 <br>
now you can install with (run cmd as admin): <br>
`python setup.py install`

## Hack to accept multiple files with streamlit api:
if you want to be able to use inference on multiple files you will have to modify the streamlit code. <br>
First find where your python side-packages are located. <br>
For Linux they are at /home/<YOUR USER>/.local/lib/<YOUR PYTHON VERSION>/site-packages/streamlit/elements/file_uploader.py <br>
Example path : /home/appuser/.local/lib/python3.6/site-packages/streamlit/elements/file_uploader.py

For Windows they are at something like "C:\Users\<YOUR USER>\AppData\Local\Programs\Python\<YOUR PYTHON VERSION>\Lib\site-packages" <br>
Example path: C:\Users\Marc\AppData\Local\Programs\Python\Python37\Lib\site-packages\streamlit

From here navigate to elements\file_uploader.py <br>
Open and change the line accept_multiple_files = False to accept_multiple_files = True <br>
This might change in future versions of streamlit to be enabled by default, but the current version I am using (0.66.0) needs this fix. <br>
I will remove this fix if any future versions of streamlit update this.