import os
import ctypes
import argparse

import numpy as np
import uff
import tensorrt as trt
import graphsurgeon as gs
import pycuda.driver as cuda

import cv2

WORK_DIR = "/LFS/mobilenetssd"
MODEL_DIR = 'models'

DEBUG_UFF = False
DIR_NAME = os.path.dirname(__file__)
LIB_FILE = os.path.abspath(os.path.join(DIR_NAME, 'installData/libflattenconcat.so'))
print ("********************",LIB_FILE)
if trt.__version__[0] < '7':
    ctypes.CDLL(LIB_FILE)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

def _preprocess_trt(img, shape=(300, 300)):
    """Preprocess an image before TRT SSD inferencing."""
    img = cv2.resize(img, shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img *= (2.0/255.0)
    img -= 1.0
    return img



def _postprocess_trt(img, output, conf_th, output_layout=7):
    """Postprocess TRT SSD output."""
    img_h, img_w, _ = img.shape
    boxes, confs, clss = [], [], []
    for prefix in range(0, len(output), output_layout):
        #index = int(output[prefix+0])
        conf = float(output[prefix+2])
        if conf < conf_th:
            continue
        x1 = int(output[prefix+3] * img_w)
        y1 = int(output[prefix+4] * img_h)
        x2 = int(output[prefix+5] * img_w)
        y2 = int(output[prefix+6] * img_h)
        cls = int(output[prefix+1])
        boxes.append((x1, y1, x2, y2))
        confs.append(conf)
        clss.append(cls)
    return boxes, confs, clss


class trtSSDEngine():
    def __init__(self,model_name='frozen_inference_graph',input_shape=(300,300),cuda_ctx=None):
        
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        import pycuda.autoinit 
        self.inputDims=(3,input_shape[0],input_shape[1])
        self.input_shape=input_shape
        model_loc = os.path.join(WORK_DIR, MODEL_DIR, model_name+'.pb')
        uff_loc=os.path.join(WORK_DIR, MODEL_DIR, model_name+'.uff')
        bin_loc=os.path.join(WORK_DIR, MODEL_DIR, model_name+'.bin')
        self.spec = {
                'input_pb':   model_loc,
                'tmp_uff':   uff_loc ,
                'output_bin': bin_loc,
                'num_classes': 91,
                'min_size': 0.2,
                'max_size': 0.95,
                'input_order': [0, 2, 1],  # order of loc_data, conf_data, priorbox_data
            }
        if not os.path.isfile(bin_loc):
            self.convert()
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()

        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()
        self.set_context()
        
        
    def replace_addv2(self,graph):
        """Replace all 'AddV2' in the graph with 'Add'.

        'AddV2' is not supported by UFF parser.

        Reference:
        1. https://github.com/jkjung-avt/tensorrt_demos/issues/113#issuecomment-629900809
        """
        for node in graph.find_nodes_by_op('AddV2'):
            gs.update_node(node, op='Add')
        return graph


    def replace_fusedbnv3(self,graph):
        """Replace all 'FusedBatchNormV3' in the graph with 'FusedBatchNorm'.

        'FusedBatchNormV3' is not supported by UFF parser.

        Reference:
        1. https://devtalk.nvidia.com/default/topic/1066445/tensorrt/tensorrt-6-0-1-tensorflow-1-14-no-conversion-function-registered-for-layer-fusedbatchnormv3-yet/post/5403567/#5403567
        2. https://github.com/jkjung-avt/tensorrt_demos/issues/76#issuecomment-607879831
        """
        for node in graph.find_nodes_by_op('FusedBatchNormV3'):
            gs.update_node(node, op='FusedBatchNorm')
        return graph


    def add_anchor_input(self,graph):
        """Add the missing const input for the GridAnchor node.

        Reference:
        1. https://www.minds.ai/post/deploying-ssd-mobilenet-v2-on-the-nvidia-jetson-and-nano-platforms
        """
        data = np.array([1, 1], dtype=np.float32)
        anchor_input = gs.create_node('AnchorInput', 'Const', value=data)
        graph.append(anchor_input)
        graph.find_nodes_by_op('GridAnchor_TRT')[0].input.insert(0, 'AnchorInput')
        return graph

    def add_plugin(self,graph,spec):
        """add_plugin

        Reference:
        1. https://github.com/AastaNV/TRT_object_detection/blob/master/config/model_ssd_mobilenet_v1_coco_2018_01_28.py
        2. https://github.com/AastaNV/TRT_object_detection/blob/master/config/model_ssd_mobilenet_v2_coco_2018_03_29.py
        3. https://devtalk.nvidia.com/default/topic/1050465/jetson-nano/how-to-write-config-py-for-converting-ssd-mobilenetv2-to-uff-format/post/5333033/#5333033
        """
        numClasses = spec['num_classes']
        minSize = spec['min_size']
        maxSize = spec['max_size']
        inputOrder = spec['input_order']

        all_assert_nodes = graph.find_nodes_by_op('Assert')
        graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)

        all_identity_nodes = graph.find_nodes_by_op('Identity')
        graph.forward_inputs(all_identity_nodes)

        Input = gs.create_plugin_node(
            name='Input',
            op='Placeholder',
            shape=(1,) + self.inputDims
        )

        PriorBox = gs.create_plugin_node(
            name='MultipleGridAnchorGenerator',
            op='GridAnchor_TRT',
            minSize=minSize,  # was 0.2
            maxSize=maxSize,  # was 0.95
            aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
            variance=[0.1, 0.1, 0.2, 0.2],
            featureMapShapes=[19, 10, 5, 3, 2, 1],
            numLayers=6
        )

        NMS = gs.create_plugin_node(
            name='NMS',
            op='NMS_TRT',
            shareLocation=1,
            varianceEncodedInTarget=0,
            backgroundLabelId=0,
            confidenceThreshold=0.3,  # was 1e-8
            nmsThreshold=0.6,
            topK=100,
            keepTopK=100,
            numClasses=numClasses,  # was 91
            inputOrder=inputOrder,
            confSigmoid=1,
            isNormalized=1
        )

        concat_priorbox = gs.create_node(
            'concat_priorbox',
            op='ConcatV2',
            axis=2
        )

        if trt.__version__[0] >= '7':
            concat_box_loc = gs.create_plugin_node(
                'concat_box_loc',
                op='FlattenConcat_TRT',
                axis=1,
                ignoreBatch=0
            )
            concat_box_conf = gs.create_plugin_node(
                'concat_box_conf',
                op='FlattenConcat_TRT',
                axis=1,
                ignoreBatch=0
            )
        else:
            concat_box_loc = gs.create_plugin_node(
                'concat_box_loc',
                op='FlattenConcat_TRT'
            )
            concat_box_conf = gs.create_plugin_node(
                'concat_box_conf',
                op='FlattenConcat_TRT'
            )

        namespace_for_removal = [
            'ToFloat',
            'image_tensor',
            'Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3',
        ]
        namespace_plugin_map = {
            'MultipleGridAnchorGenerator': PriorBox,
            'Postprocessor': NMS,
            'Preprocessor': Input,
            'ToFloat': Input,
            'Cast': Input,  # added for models trained with tf 1.15+
            'image_tensor': Input,
            'MultipleGridAnchorGenerator/Concatenate': concat_priorbox,  # for 'ssd_mobilenet_v1_coco'
            'Concatenate': concat_priorbox,  # for other models
            'concat': concat_box_loc,
            'concat_1': concat_box_conf
        }

        graph.remove(graph.find_nodes_by_path(['Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3']), remove_exclusive_dependencies=False)  # for 'ssd_inception_v2_coco'

        graph.collapse_namespaces(namespace_plugin_map)
        graph = self.replace_addv2(graph)
        graph = self.replace_fusedbnv3(graph)

        if 'image_tensor:0' in graph.find_nodes_by_name('Input')[0].input:
            graph.find_nodes_by_name('Input')[0].input.remove('image_tensor:0')
        if 'Input' in graph.find_nodes_by_name('NMS')[0].input:
            graph.find_nodes_by_name('NMS')[0].input.remove('Input')
        # Remove the Squeeze to avoid "Assertion 'isPlugin(layerName)' failed"
        graph.forward_inputs(graph.find_node_inputs_by_name(graph.graph_outputs[0], 'Squeeze'))
        if 'anchors' in [node.name for node in graph.graph_outputs]:
            graph.remove('anchors', remove_exclusive_dependencies=False)
        if len(graph.find_nodes_by_op('GridAnchor_TRT')[0].input) < 1:
            graph = add_anchor_input(graph)
        if 'NMS' not in [node.name for node in graph.graph_outputs]:
            graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)
            if 'NMS' not in [node.name for node in graph.graph_outputs]:
                # We expect 'NMS' to be one of the outputs
                raise RuntimeError('bad graph_outputs')

        return graph
               
    def convert(self):
        dynamic_graph = self.add_plugin(
            gs.DynamicGraph(self.spec['input_pb']),
            self.spec)
        
        _ = uff.from_tensorflow(
            dynamic_graph.as_graph_def(),
            output_nodes=['NMS'],
            output_filename=self.spec['tmp_uff'],
            text=True,
            debug_mode=DEBUG_UFF)
        
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
            builder.max_workspace_size = 1 << 28
            builder.max_batch_size = 1
            builder.fp16_mode = True

            parser.register_input('Input', self.inputDims)
            parser.register_output('MarkOutput_0')
            parser.parse(self.spec['tmp_uff'], network)
            engine = builder.build_cuda_engine(network)

            buf = engine.serialize()
            with open(self.spec['output_bin'], 'wb') as f:
                f.write(buf)

    def _load_engine(self):
        TRTbin = self.spec['output_bin']
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings = \
            [], [], [], [], []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
        return host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings
    
    def set_context(self,cuda_ctx=None):
        try:
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.host_inputs, self.host_outputs, self.cuda_inputs, self.cuda_outputs, self.bindings = self._allocate_buffers()
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e

    def __del__(self):
        """Free CUDA memories and context."""
        del self.cuda_outputs
        del self.cuda_inputs
        del self.stream
        
    def detect(self, img, conf_th=0.3):
        """Detect objects in the input image."""
        img_resized = _preprocess_trt(img, self.input_shape)
        np.copyto(self.host_inputs[0], img_resized.ravel())

        if self.cuda_ctx:
            self.cuda_ctx.push()
        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(
            self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        output = self.host_outputs[0]
        return _postprocess_trt(img, output, conf_th)
            
            
    
        
