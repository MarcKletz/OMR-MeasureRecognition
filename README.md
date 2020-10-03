# THIS REPO AND README IS STILL UNDER HEAVY DEVELOPMENT!

# About this repo

All data required for this GitHub project can be downloaded by using the provided DaterLoader script.  
Simply calling:  
```
DataLoader().download_datasets(root_dir)
DataLoader().download_trained_models(root_dir)
DataLoader().generateAllJsonDataAnnotations(root_dir)
``` 
Will download the datasets from the OMR-Datasets repository,  
download all the trained models as specified in the overview and  
generate the json annotations for the datasets.

## Faster R-CNN with ResNet-50 backbone
|   Model Name        					     |   Iterations  |   mAP    |   AP75   |   AP50   | system measures mAP |  staves mAP   |  stave measures mAP   |
|:------------------------------------------:|:-------------:|:--------:|:--------:|:--------:|:-------------------:|:-------------:|:---------------------:|
|   System measures   					     |     16500     |  96.547  |  98.935  |  98.970  |        XX           |      XX       |           XX          |
|   Stave measures    					     |     16200     |  88.168  |  96.860  |  98.013  |        XX           |      XX       |           XX          |
|      Staves         					     |     10800     |  93.596  |  98.987  |  100.00  |        XX           |      XX       |           XX          |
| System measures and Staves                 |     15000     |  90.792  |  96.481  |  96.500  |       95.532        |    86.053     |           XX          |
| System measures, Stave measures and Staves |     4500      |  77.779  |  86.884  |  87.404  |       90.010        |    78.622     |         64.706        |



## Faster R-CNN with ResNet-101 backbone
|   Model Name        					     |   Iterations  |   mAP    |   AP75   |   AP50   | system measures mAP |  staves mAP   |  stave measures mAP   |
|:------------------------------------------:|:-------------:|:--------:|:--------:|:--------:|:-------------------:|:-------------:|:---------------------:|
|   System measures   					     |     19500     |  97.112  |  98.928  |  98.949  |        XX           |      XX       |           XX          |
|   Stave measures    					     |     17100     |  89.254  |  97.903  |  98.018  |        XX           |      XX       |           XX          |
|      Staves         					     |     18000     |  94.004  |  99.010  |  100.00  |        XX           |      XX       |           XX          |
| System measures and Staves                 |     18300     |  91.301  |  96.478  |  96.498  |       95.768        |    86.834     |           XX          |
| System measures, Stave measures and Staves |     2700      |  77.829  |  88.697  |  89.366  |       85.383        |    79.779     |         68.324        |


<!-- STILL NEEDS TO BE TRAINED
## Faster R-CNN with ResNeXt-101-32x8d backbone
|   Model Name        					     |   Iterations  |   mAP    |   AP75   |   AP50   | system measures mAP |  staves mAP   |  stave measures mAP   |
|:------------------------------------------:|:-------------:|:--------:|:--------:|:--------:|:-------------------:|:-------------:|:---------------------:|
|   System measures   					     |     19500     |  97.112  |  98.928  |  98.949  |        XX           |      XX       |           XX          |
|   Stave measures    					     |     17100     |  89.254  |  97.903  |  98.018  |        XX           |      XX       |           XX          |
|      Staves         					     |     10800     |  93.596  |  98.987  |  100.00  |        XX           |      XX       |           XX          |
| System measures and Staves                 |     18300     |  91.301  |  96.478  |  96.498  |       95.768        |    86.834     |           XX          |
| System measures, Stave measures and Staves |     2700      |  77.829  |  88.697  |  89.366  |       85.383        |    79.779     |         68.324        |
-->

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
Dont forget to remove the opencv-python-headless requirement from the linux_requirements.txt if you did this!  

Step 4:  
install all the required python libraries from this repository:  
```sudo pip3 install -r Python/linux_requirements.txt [-v]```  
This might take a while! So be patient, you may add the -v tag to see installation progress.  

## For Windows:

Requirements:  
Windows SDK  
C++14.0 build tools  
Microsoft Visual C++ Redistributable  
can all be installed with the Visual Studio installer.  
https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16  
![](Images/VS_setup.png)

Step 1:  
install all the required python libraries from the OMR-MeasureRecognition repo.  
```pip install -r Python/windows_requirements.txt```

Step 2:  
manually install detectron2 because there is no windows support, we need to pull and install from source  
```
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
git reset --hard be792b959bca9af0aacfa04799537856c7a92802 # to pull detectron version 0.2.1
```

Step 3:  
change the following line in detectron2\detectron2\layers\csrc\cocoeval\cocoeval.cpp(483):  
localtime_r(&rawtime, &local_time) to localtime_s(&local_time, &rawtime);  
solution from : https://github.com/conansherry/detectron2/issues/2  

Step 4:  
now you can install detectron2 with:  
```python setup.py install```
Requires admin privileges, so run cmd as admin!

# Run the Streamlit app:

Make sure that the python package installation location is added to path, so that you can run streamlit. If the streamlit command fails with "command not found" you will need to add the following to your path:  
```export PATH="$HOME/.local/bin:$PATH"```

Complete the installation instructions and then run:  
```streamlit run Python/streamlit_app.py```  
from the OMR-MeasureRecognition repository

## Hack to accept multiple files with streamlit API:

if you want to be able to use inference on multiple files you will have to modify the streamlit code.  
First find where your python side-packages are located by running:
```pip show streamlit```  
From here navigate to elements\file_uploader.py  
Open and change the line accept_multiple_files = False to accept_multiple_files = True  
This might change in future versions of streamlit to be enabled by default, but the current version I am using (0.66.0) needs this fix.  
I will remove this fix if any future versions of streamlit update this.