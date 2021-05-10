# README #

Simple object detection library using mobilenet ssd arachitectures.

### Supports ###

* Tensorflow cpu support
* Tensorflow gpu support
* TF-TRT support(nvidia tensorflow docker required)
* Native Tensorrt support(nvidia tensorflow docker required)(Tensorrt version 7+)
* Auto download object and face models
* Support for L4T containers


### Pre requisite Requirements ###
* Tensorflow 1.X
* Opencv

### Installation ###
```sh
pip3 install git+https://github.com/MacherLabs/objectDetectorSSD.git@tensorrt
```
### How to use ###
```
#Import
import numpy as np
from objectDetectionSSD import ObjectDetectorSSD

#Prepare inputes
model_name='frozen_inference_graph.pb' #Frozen pb graph
trt_enable=True # TF-TRT conversion enablement
tensorrt_enable=False #Tensorrt conversion enablement
precision='FP16' # Precision to convert to
classes=['person','car'] # Classes to predict(if [], predicts all classes in labels.json)
thresh=0.5 # Threshold for detection confidence
input_shape=300 # Input shape of model
gpu_frac = #Amount of gpu to allocate, if 0 allows growth

#Initialize detector
detector=ObjectDetectorSSD(model_name=model_name,
                        gpu_frac=gpu_frac,
                        tf_trt_enable=trt_enable,
                        tensorrt_enable=tensorrt_enable,
                        precision=precision,
                        classes=classes,
                        threshold=thresh,
                        input_size=input_shape
                    )
# Generate test image
img =np.random.rand(300,300,3).astype('uint8')

# Perform detections
dets=detector.detect(img,threshold=thresh)

```
 
 ### Benchmarking on jetson nano ###
 Mobilenet_300_ssd- 17 fps <br>
 Mobilenet_512_ssd- 9 fps
 
 
 
