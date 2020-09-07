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

General prerequirements
Python >= 3.6
Git

## For Linux:

simply run the command:
pip install -r linux_requirements.txt

## For Windows:

Prereqs: <br>
Windows SDK <br>
C++14 compiler <br>

git clone https://github.com/facebookresearch/detectron2.git <br>
git reset --hard be792b959bca9af0aacfa04799537856c7a92802 # pull detectron 0.2.1 <br>
cd into the detectron2 folder where setup.py is <br>
python setup.py install

ERROR: <br>
detectron2\layers\csrc\cocoeval\cocoeval.cpp(483): error C3861: 'localtime_r': identifier not found <br>
https://github.com/conansherry/detectron2/issues/2

change localtime_r(a, b) to localtime_s(&local_time, &rawtime); <br>
save and run  <br>
python setup.py install again

detectron2/layers/csrc/nms_rotated/nms_rotated_cuda.cu(14): error: name must be a namespace name <br>
insert #define WITH_HIP before #ifdef WITH_HIP

ERROR: <br>
cl : Command line error D8021 : invalid numeric argument '/Wno-cpp' <br>
error: command 'C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\BIN\\x86_amd64\\cl.exe' failed with exit status 2

pull cocodataset from <br>
https://github.com/cocodataset/cocoapi

the solution is here by user TonyNgo1 (real MVP): <br>
https://github.com/cocodataset/cocoapi/issues/51

cd into the folder and run <br>
python setup.py install

now cd back to detectron2 and complete <br>
python setup.py install