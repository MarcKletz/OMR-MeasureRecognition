# THIS REPO AND README IS STILL UNDER HEAVY DEVELOPMENT!

# About this repo
## [This repo has a live app running here](https://share.streamlit.io/marckletz/omr-measurerecognition/Python/streamlit_app.py)


## Faster R-CNN with ResNet-50 backbone
|   Model Name        					     |   Iterations  |   mAP    |   AP75   |   AP50   | system measures mAP |  staves mAP   |  stave measures mAP   |
|:------------------------------------------:|:-------------:|:--------:|:--------:|:--------:|:-------------------:|:-------------:|:---------------------:|
|   System measures   					     |     16500     |  96.352  |  99.010  |  99.010  |        XX           |      XX       |           XX          |
|   Stave measures    					     |     16200     |  88.003  |  97.791  |  98.020  |        XX           |      XX       |           XX          |
|      Staves         					     |     10800     |  93.387  |  100.00  |  100.00  |        XX           |      XX       |           XX          |
| System measures and Staves                 |     15000     |  90.746  |  96.526  |  96.532  |       95.487        |    86.004     |           XX          |
| System measures, Stave measures and Staves |     4500      |  77.485  |  86.905  |  87.103  |       89.909        |    78.344     |         64.202        |



## Faster R-CNN with ResNet-101 backbone
|   Model Name        					     |   Iterations  |   mAP    |   AP75   |   AP50   | system measures mAP |  staves mAP   |  stave measures mAP   |
|:------------------------------------------:|:-------------:|:--------:|:--------:|:--------:|:-------------------:|:-------------:|:---------------------:|
|   System measures   					     |     19500     |  96.995  |  99.010  |  99.010  |        XX           |      XX       |           XX          |
|   Stave measures    					     |     17100     |  88.997  |  97.885  |  98.020  |        XX           |      XX       |           XX          |
|      Staves         					     |     18000     |  93.978  |  99.008  |  100.00  |        XX           |      XX       |           XX          |
| System measures and Staves                 |     18300     |  91.040  |  96.529  |  96.530  |       95.564        |    86.516     |           XX          |
| System measures, Stave measures and Staves |     2700      |  77.404  |  88.382  |  89.080  |       84.701        |    79.450     |         68.064        |



## Faster R-CNN with ResNeXt-101-32x8d backbone
|   Model Name        					     |   Iterations  |   mAP    |   AP75   |   AP50   | system measures mAP |  staves mAP   |  stave measures mAP   |
|:------------------------------------------:|:-------------:|:--------:|:--------:|:--------:|:-------------------:|:-------------:|:---------------------:|
|   System measures   					     |     18600     |  96.751  |  99.002  |  99.002  |        XX           |      XX       |           XX          |
|   Stave measures    					     |     18300     |  89.058  |  97.883  |  97.999  |        XX           |      XX       |           XX          |
|      Staves         					     |     9900      |  94.171  |  100.00  |  100.00  |        XX           |      XX       |           XX          |
| System measures and Staves                 |     20000     |  91.221  |  96.521  |  96.527  |       96.816        |    85.625     |           XX          |
| System measures, Stave measures and Staves |     4200      |  76.546  |  86.203  |  86.785  |       84.638        |    72.900     |         72.100        |

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
Dont forget to remove the opencv-python-headless requirement from the linux_requirements.txt if you did this!  

**Debian:**  
Skip to Step 4  

Step 4:  
install all the required python libraries from this repository:  
```sudo pip3 install -r requirements.txt [-v]```  
This might take a while! So be patient, you may add the -v tag to see installation progress.  

Step 5:  
Install the Detectron2 submodule as python library by running  
```python setup.py install```  
from within the Detectron2 folder.  
(Requires admin privileges, so run cmd as admin!)

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
```pip install -r requirements.txt```

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