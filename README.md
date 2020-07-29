# Computer Pointer Controller

This pointer Controller app controls the movement of mouse pointer by detecting the direction of eyes and also from the pose of head. This app takes video as input and then app estimates eye-direction and head-pose and based on that estimation it move the mouse pointers.

Openvino Version: 2020.1
Python version: 3.6.9

Models Required:
---------------------------------------------
1. [Face Detection Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html) 
2. [Facial Landmark Detection Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
3. [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
4. [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)


## Project Set Up and Installation

1. For running this application you need to install openvino successfully. For installing Openvino on your OS operating system you can go through [this](https://docs.openvinotoolkit.org/latest/index.html). 
2. Clone this repository. i. e.  https://github.com/wagle1996/Computer_pointer_controller
3. Initialize openvino environment on your system by pasting this command on the terminal
'''
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
'''
4. Install dependencies using the command 
'''
pip3 install -r requirements.txt
'''
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
'''
5. Downlaod and Install 4 models from openvino before running this:

**1. <B> Face detection model</B>**
```
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"
```
**2. <B> Facial landmarks detection model </B>**
```
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
```
**3.<B> Head pose Estimation Model </B>**
```
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"
```
**4.<B> Gaze Estimation Model </B>**
```
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
```
## Directory Structure
```
.
├── bin
│   └── demo.mp4
├── intel
│   ├── face-detection-adas-binary-0001
│   │   └── FP32-INT1
│   │       ├── face-detection-adas-binary-0001.bin
│   │       └── face-detection-adas-binary-0001.xml
│   ├── face-detection-retail-0004
│   │   └── FP16
│   │       ├── face-detection-retail-0004.bin
│   │       └── face-detection-retail-0004.xml
│   ├── gaze-estimation-adas-0002
│   │   ├── FP16
│   │   │   ├── gaze-estimation-adas-0002.bin
│   │   │   └── gaze-estimation-adas-0002.xml
│   │   ├── FP32
│   │   │   ├── gaze-estimation-adas-0002.bin
│   │   │   └── gaze-estimation-adas-0002.xml
│   │   └── FP32-INT8
│   │       ├── gaze-estimation-adas-0002.bin
│   │       └── gaze-estimation-adas-0002.xml
│   ├── head-pose-estimation-adas-0001
│   │   ├── FP16
│   │   │   ├── head-pose-estimation-adas-0001.bin
│   │   │   └── head-pose-estimation-adas-0001.xml
│   │   ├── FP32
│   │   │   ├── head-pose-estimation-adas-0001.bin
│   │   │   └── head-pose-estimation-adas-0001.xml
│   │   └── FP32-INT8
│   │       ├── head-pose-estimation-adas-0001.bin
│   │       └── head-pose-estimation-adas-0001.xml
│   └── landmarks-regression-retail-0009
│       ├── FP16
│       │   ├── landmarks-regression-retail-0009.bin
│       │   └── landmarks-regression-retail-0009.xml
│       ├── FP32
│       │   ├── landmarks-regression-retail-0009.bin
│       │   └── landmarks-regression-retail-0009.xml
│       └── FP32-INT8
│           ├── landmarks-regression-retail-0009.bin
│           └── landmarks-regression-retail-0009.xml
├── README.md
├── requirements.txt
└── src
    ├── face_detection.py
    ├── facial_landmarks_detection.py
    ├── gaze_estimation.py
    ├── head_pose_estimation.py
    ├── input_feeder.py
    ├── main.py
    ├── mouse_controller.py
    └── __pycache__
        ├── face_detection.cpython-36.pyc
        ├── face_detection.py
        ├── facial_landmarks_detection.cpython-36.pyc
        ├── gaze_estimation.cpython-36.pyc
        ├── head_pose_estimation.cpython-36.pyc
        ├── input_feeder.cpython-36.pyc
        └── mouse_controller.cpython-36.pyc

```
6. In the terminal go the project file folder

```
`cd <project-repo-path>/src``

```
7. Run the command


**1. For CPU**
```
python3 main.py -fd ../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fld ../intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -hpe ../intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -ge ../intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i../bin/demo.mp4 -d CPU -flags fdm lrm hp gze 4

```
## Demo

After you run the command you will get the following result <BR>
Click on the picture below to see the demo video

[![Demo video](https://i9.ytimg.com/vi/9O4Z9yyVdzQ/mq2.jpg?sqp=CIyt8PgF&rs=AOn4CLAmNtZOwUr1BS1sJtkdLRbplHdFGA)](https://youtu.be/9O4Z9yyVdzQ)

## Documentation
```
usage: main.py [-h] -fd FACE_DETECTION_MODEL -fld
               FACIAL_LANDMARKS_DETECTION_MODEL -hpe
               HEAD_POSE_ESTIMATION_MODEL -ge GAZE_ESTIMATION_MODEL -i INPUT
               [-d DEVICE] [-l CPU_EXTENSION] [-pt PROB_THRESHOLD]
               [-flags OUTPUT_FLAGS [OUTPUT_FLAGS ...]]


optional arguments:
  -h, --help            show this help message and exit
  -fd FACE_DETECTION_MODEL, --face_detection_model FACE_DETECTION_MODEL
                        Path to an .xml file with Face Detection trained
                        model.
  -fld FACIAL_LANDMARKS_DETECTION_MODEL, --facial_landmarks_detection_model FACIAL_LANDMARKS_DETECTION_MODEL
                        Path to an .xml file with Facial Landmark Detection
                        trained model.
  -hpe HEAD_POSE_ESTIMATION_MODEL, --head_pose_estimation_model HEAD_POSE_ESTIMATION_MODEL
                        Path to an .xml file with Head Pose Estimation trained
                        model.
  -ge GAZE_ESTIMATION_MODEL, --gaze_estimation_model GAZE_ESTIMATION_MODEL
                        Path to an .xml file with Gaze Estimation trained
                        model.
  -i INPUT, --input INPUT
                        Path to image or video file or CAM
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample will look for a
                        suitable plugin for device specified (CPU by default)
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with thekernels impl.
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for detection fitering.
  -flags OUTPUT_FLAGS [OUTPUT_FLAGS ...], --output_flags OUTPUT_FLAGS [OUTPUT_FLAGS ...]
                        Specify flag for visualization for different
                        model(Space separated if multiple values)fdm for
                        faceDetectionModel, lrm for landmarkRegressionModelhp
                        for headPoseEstimationModel, gze for
                        gazeEstimationModel
```

## Benchmarks
I run the model on processor Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz on brigde Xeon E3-1200 v6/7th Gen Core Processor using 3 different model precisions:

|Model Precision |Total Inference time   |total model Loading time|
|----------------|-----------------------|------------------------|
|FP32-INT8      |426.13211728758733 ms   |20.57350288003178 ms   |
|FP32           |387.99005443766964 ms   |5.6814420021186445 ms  |
|FP16           |389.5139815443653 ms    |6.439516099832826 ms   |

## Results
As shown in the table above, performance metrics are pretty much the same for all precisions tested, with load time and inference time very much higher in FP32-INT8 precision. Accuracy was not affected by changing the precision, perceived stability of the pointer position was pretty much identical for all tests. The size of FP16 presicion model is much smaller than other model plus the model loading time and total model inference time is less than other models so, I suggest to use FP16 model.
