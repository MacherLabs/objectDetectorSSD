import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Loaded " + __name__)

import os
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import tensorRTEngine as ten
#import pandas as pd
import numpy as np
import json
import cv2

WORK_DIR = "/LFS/mobilenetssd"
MODEL_DIR = 'models'
MODEL_URLS=['http://api.staging.vedalabs.in/filemanager/7VLnJ/mobilenet_512_frozen_inference_graph_face.pb',
            'http://api.staging.vedalabs.in/filemanager/hSpFV/mobilenet_300_frozen_inference_graph.pb']




class ObjectDetectorSSD():
    
    def __init__(self,model_name=None,precision ='FP32',input_size=300,threshold=0.3,gpu_frac=0,tf_trt_enable=False,tensorrt_enable=False,classes=['person'],):
        
        self.threshold = threshold
        self.classes=classes
        self.tensorrt_enable=tensorrt_enable
        label_path=os.path.abspath(os.path.dirname(__file__))
        with open('{}/labels.json'.format(label_path)) as f:
            self.labels=json.load(f)
            print("labels",self.labels)
            
        if model_name==None:
            model_name='mobilenet_{}_frozen_inference_graph.pb'.format(input_size)
            
        model_loc = os.path.join(WORK_DIR, MODEL_DIR, model_name)
        
        if not os.path.isfile(model_loc):
            try:
                os.makedirs(os.path.join(WORK_DIR, MODEL_DIR))
            except:
                pass
            self.download_model(model_name)

        if tensorrt_enable==True:
            self.detector=ten.trtSSDEngine(model_name=model_name[:-3],input_shape=(input_size,input_size))

        else:  
        
            if trt_enable ==True:
                trt_model_name="{}_{}_{}".format("trt",precision,model_name)
                trt_loc=os.path.join(WORK_DIR, MODEL_DIR, trt_model_name)
                
                logger.info("tensorrt graph location-{}".format(trt_loc))
                if os.path.isfile(trt_loc):
                    logger.info("found converted trt graph..no conversion needed")
                    model_loc=trt_loc
                    
            self.detection_graph = tf.Graph()

            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(model_loc, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    # Convert to tensorrt graph if asked
                    if trt_enable == True and model_loc != trt_loc:
                        logger.info("converting to trt-graph,please wait..")
                        trt_graph_def=trt.create_inference_graph(input_graph_def= od_graph_def,
                                                    max_batch_size=1,
                                                    max_workspace_size_bytes=1<<20,
                                                    precision_mode=precision,
                                                    minimum_segment_size=5,
                                                    maximum_cached_engines=5,
                                                    outputs=['detection_boxes','detection_scores','detection_classes','num_detections'])
                        logger.info("conversion to trt graph completed")
                        with tf.gfile.GFile(trt_loc, "wb") as f:
                            f.write(trt_graph_def.SerializeToString())
                        logger.info("saved trt graph to {}".format(trt_loc))
                            
                        tf.import_graph_def(trt_graph_def, name='')
                    else:
                        tf.import_graph_def(od_graph_def, name='')

            with self.detection_graph.as_default():
                config = tf.ConfigProto()
                if gpu_frac == 0:
                    config.gpu_options.allow_growth = True
                else:
                    config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
                self.sess = tf.Session(graph=self.detection_graph, config=config)
                self.windowNotSet = True
          
    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """
        image_np = image#cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        #start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        #elapsed_time = time.time() - start_time
        #print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)
    
#   Detect objects
    def detect(self, imgcv, **kwargs):
        threshold = kwargs.get('threshold', self.threshold)
        if self.tensorrt_enable == True:
            results=self.detector.detect(imgcv,threshold)
            return self._format_result_tensorrt(results)
        else:
            results = self.run(imgcv)
        im_height,im_width,_ = imgcv.shape
        return self._format_result(results,threshold,im_width,im_height)
    
    def _format_result_tensorrt(self,result):
        out_list=[]
        boxes,scores,classes=result
        num_detections=len(boxes)
        for index in range(int(num_detections)):
            clas=str(self.labels[str(int(classes[index]))])
            
            if clas in self.classes or len(self.classes)==0:
                box = boxes[index]
                xmin,ymin, xmax, ymax = box[0],box[1],box[2],box[3]
                (left, right, top, bottom) = (xmin,xmax,ymin,ymax )
                prob = scores[index]
                formatted_res = dict()
                formatted_res["class"] = clas
                formatted_res["prob"] = prob
                formatted_res["box"] = {
                    "topleft": {'x': int(left), 'y': int(top)},
                    "bottomright": {'x': int(right), 'y': int(bottom)}
                    }
                out_list.append(formatted_res)
        #Return the result
        return out_list

    # Format the results
    def _format_result(self, result,threshold,im_width,im_height):
        out_list = []
        boxes,scores,classes,num_detections = result
        boxes = boxes[0] # index by 0 to remove batch dimension
        scores = scores[0]
        classes = classes[0]
        num_detections = num_detections[0]
        indexes = np.squeeze(np.argwhere(scores>threshold),axis=1)
        #print("indexes",indexes)
        for index in range(int(num_detections)):
            clas=str(self.labels[str(int(classes[index]))])
            
            if clas in self.classes or len(self.classes)==0:
                box = boxes[index]
                ymin, xmin, ymax, xmax = box[0],box[1],box[2],box[3]
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
                prob = scores[index]
                if prob> threshold:
                    formatted_res = dict()
                    formatted_res["class"] = clas
                    formatted_res["prob"] = prob
                    formatted_res["box"] = {
                        "topleft": {'x': int(left), 'y': int(top)},
                        "bottomright": {'x': int(right), 'y': int(bottom)}
                        }
                    out_list.append(formatted_res)
        #Return the result
        return out_list
    
    def download_model(self,model):
        if '512' in model:
            url = MODEL_URLS[0]
        else:
            url = MODEL_URLS[1]
        download_location = os.path.join(WORK_DIR, MODEL_DIR)
        cmd = 'cd {};wget {}'.format(download_location,url)
        os.system(cmd)
        
    def draw_rects(self,img, faces):
        """
        Draws rectangle around detected faces.
        Arguments:
            img: image in numpy array on which the rectangles are to be drawn
            faces: list of faces in a format given in Face Class
        Returns:
            img: image in numpy array format with drawn rectangles
        """
        for face in faces:
            x1, y1 = face['box']['topleft']['x'], face['box']['topleft']['y']
            x2, y2 = face['box']['bottomright']['x'], face['box']['bottomright']['y']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return img
            
        
if __name__ == '__main__':
    import sys
    import pprint
    import cv2

    detector = ObjectDetectorSSD(tensorrt_enable=True)
    image_url = 'test.jpg'
    imgcv = cv2.imread(image_url)
    if imgcv is not None:
        print(imgcv.shape)
        results = detector.detect(imgcv)
        pprint.pprint(results)
        im = detector.draw_rects(imgcv, results)
        cv2.imwrite("result.jpg",im)
#        cv2.imshow('Faces', draw_rects(imgcv, results))
#        cv2.waitKey(5000)
    else:
        print("Could not read image: {}".format(image_url))
        
        
            
        
        
        
        

