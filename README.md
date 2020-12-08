# About this repo [![This repo has a live app running here](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/marckletz/omr-measurerecognition/Python/streamlit_app.py)

This repository has been created as part of my master's thesis. The thesis focuses on detecting system measures, stave measures and staves in music pages with the help of deep learning. I used two open-source datasets for this thesis. The handwritten MUSCIMA++ dataset, which has all three categories and the typeset bounding box annotations of musical measures dataset, which only contains system measures. I first enhanced the typeset dataset by using a deep learning framework called Detectron2. This has been achieved by training a model on only the stave measures from the MUSCIMA++ dataset. The trained model generalized well on the typeset data, so that I could predict stave measures from it. I took the stave measures from the predictions, created staves from them and intersected them with the existing ground truth system measures to generate more precise stave measures. After manually correcting the generated stave measure annotations, I generated stave lines from the system measures and stave measures. These have also been manually corrected where needed. With both datasets now containing all three categories, I trained several models, one for each category, one for system measures and staves, and one for all three categories combined. I also used three different backbones to compare results between them. Our results show that the models generalize well for typeset music pages, but several flaws for handwritten pages have been observed. A web-based tool has been developed and deployed to use our models which is publicly available.

With this work, I achieved remarkable results for object detection in music notes. They are state-of-the-art results and the models produced can be used by anyone by visiting the live Streamlit application or cloning, installing all dependencies and running the application locally on their own pc. To run locally, follow the installation instructions below. Running locally will download all our pre-trained models which are 8GB large.  

All scripts used for this work are in the Python folder. The [training](https://github.com/MarcKletz/OMR-MeasureRecognition/blob/master/Python/training.py), [inference](https://github.com/MarcKletz/OMR-MeasureRecognition/blob/master/Python/inference.py) and [evaluation](https://github.com/MarcKletz/OMR-MeasureRecognition/blob/master/Python/evaluation.py) scripts are probably the most important ones. The scripts demonstrate how I trained all the networks, used inference to manually check their accuracy on data and how I evaluated them based on the COCO-metric. The training process how we enhanced the typeset dataset is also available as a Jupyter Notebook in the [AudioLabsEnhancement](https://github.com/MarcKletz/OMR-MeasureRecognition/blob/master/Python/AudioLabsEnhancement.ipynb) file. Other scripts are used as classes to bundle functionality together. The Streamlit application runs from the [streamlit_app](https://github.com/MarcKletz/OMR-MeasureRecognition/blob/master/Python/streamlit_app.py) file.

There are two different requirement files because the requirements.txt file is used by the Streamlit application.  
It is different from the local_requirements.txt file in that it does not use the integrated submodule because Streamlit does not yet support submodules.  
This will be changed ASAP when the support for submodules is implemented by the Streamlit framework.

# Results  

## Faster R-CNN with ResNet-50 backbone
|   Category Name        					     |   Iterations  |   mAP    |   AP75   |   AP50   | system measures mAP |  staves mAP   |  stave measures mAP   |
|:----------------------------------------------:|:-------------:|:--------:|:--------:|:--------:|:-------------------:|:-------------:|:---------------------:|
|   System measures   					     	 |     12600     |  95.828  |  98.785  |  98.982  |        -            |      -        |           -           |
|   Stave measures   					     	 |     12900     |  87.639  |  97.582  |  98.933  |        -            |      -        |           -           |
|   Staves   					     			 |     16500     |  92.578  |  99.003  |  99.010  |        -            |      -        |           -           |
|   System measures and Staves   				 |     14100     |  88.190  |  95.423  |  95.519  |        93.668       |      82.711   |           -           |
|   System measures, Stave measures and Staves   |      3600     |  75.970  |  85.549  |  86.422  |        83.366       |      79.535   |           65.010      |



## Faster R-CNN with ResNet-101 backbone
|   Category Name        					     |   Iterations  |   mAP    |   AP75   |   AP50   | system measures mAP |  staves mAP   |  stave measures mAP   |
|:----------------------------------------------:|:-------------:|:--------:|:--------:|:--------:|:-------------------:|:-------------:|:---------------------:|
|   System measures   					         |     15600     |  95.996  |  98.823  |  98.988  |        -            |      -        |           -           |
|   Stave measures   					         |     12600     |  88.882  |  97.515  |  98.938  |        -            |      -        |           -           |
|   Staves   					     		     |     19200     |  93.650  |  100.00  |  100.00  |        -            |      -        |           -           |
|   System measures and Staves   			     |      5400     |  88.886  |  96.962  |  97.018  |        93.651       |      84.122   |           -           |
|   System measures, Stave measures and Staves   |      3000     |  75.041  |  85.297  |  86.713  |        85.676       |      78.454   |           60.992      |



## Faster R-CNN with ResNeXt-101-32x8d backbone
|   Category Name        					     |   Iterations  |   mAP    |   AP75   |   AP50   | system measures mAP |  staves mAP   |  stave measures mAP   |
|:----------------------------------------------:|:-------------:|:--------:|:--------:|:--------:|:-------------------:|:-------------:|:---------------------:|
|   System measures   					         |      8400     |  95.907  |  98.931  |  99.008  |        -            |      -        |           -           |
|   Stave measures   					         |     15300     |  89.625  |  97.785  |  99.001  |        -            |      -        |           -           |
|   Staves   					     			 |     10800     |  93.457  |  99.009  |  100.00  |        -            |      -        |           -           |
|   System measures and Staves   			     |     16800     |  88.941  |  95.319  |  95.693  |        93.792       |      84.091   |           -           |
|   System measures, Stave measures and Staves   |      1800     |  75.922  |  86.017  |  87.059  |        90.096       |      77.275   |           60.393      |

# Cloning this repository  
This repository uses Detectron2 as submodule.  
In order to clone the submodule correctly, you will need to use:  
```
git clone --recurse-submodules https://github.com/MarcKletz/OMR-MeasureRecognition
```

If you already cloned the project and forgot --recurse-submodules,  
you can combine the git submodule init and git submodule update steps by running  
```
git submodule update --init
```

# Installation Setup

Requirements before starting:  
Python >= 3.6  
to run training and testing you need a CUDA capable device and the CUDA Toolkit 10.1  
you can run the streamlit app which does inference without CUDA

## For Linux:

Step 1:  
You will require some build / development tools, install them by running:  
```
sudo yum groupinstall "Development Tools"
or
sudo apt-get install build-essential
```

Step 2:  
Install python development version.  
```
sudo yum install python36-devel
or
sudo apt-get install python3-dev
```

Step 3 (OS DEPENDENT):  
**CentOS, Amazon Linux AMI, Red Hat Enterprise Linux:**  
Needs cython before running the requirements install:  
```pip3 install cython```  
This is needed for pycocotools because pip apparently builds all packages first, before attempting to install them.  
(ﾉ☉ヮ⚆)ﾉ ┻━┻

**Ubuntu:**  
There are no wheels available for opencv-python-headless on some ubuntu distributions.  
Instead of building it on your own, I recommend to install it with the following command.  
```sudo apt install python3-opencv```  
Dont forget to remove the opencv-python-headless requirement from the local_requirements.txt if you did this!  

**Debian:**  
Skip to Step 4  

Step 4:  
install all the required python libraries from this repository:  
```sudo pip3 install -r local_requirements.txt [-v]```  
This might take a while! So be patient, you may add the -v tag to see installation progress.  

Step 5:  
Install the Detectron2 submodule as python library by running  
```sudo python setup.py install```  
from within the Detectron2 folder.  

## For Windows:

Requirements:  
Windows SDK  
C++14.0 build tools  
Microsoft Visual C++ Redistributable  
can all be installed with the [Visual Studio installer](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16).
![](Images/VS_setup.png)

Step 1:  
install all the required python libraries from the OMR-MeasureRecognition repo.  
```pip install -r local_requirements.txt```  
(Requires admin privileges!)  

Step 2:  
Install the Detectron2 submodule as python library by running  
```python setup.py install```  
from within the Detectron2 folder.  
(Requires admin privileges, so run cmd as admin!)

Possible step 3:  
If step 3 fails with an error message about an nms_rotated_cuda.cu file, try this.  
add the following line in detectron2\detectron2\layers\csrc\nms_rotated\nms_rotated_cuda.cu before #ifdef WITH_HIP:  
#define WITH_HIP  
Repead step 2.


# Run the Streamlit app:
Make sure that the python package installation location is added to path, so that you can run streamlit. If the streamlit command fails with "command not found" you will need to add the following to your path:  
```export PATH="$HOME/.local/bin:$PATH"```

Complete the installation instructions and then run:  
```streamlit run Python/streamlit_app.py```  
from the OMR-MeasureRecognition repository
