import time
from itertools import product
import math
from math import sqrt
import cv2
import numpy as np
import onnxruntime
from multiprocessing import Process, Queue

from utils.box_utils import nms_numpy, after_nms_numpy
from utils.metrics_utils import APDataObject, prep_metrics

MASK_SHAPE = (138, 138, 3)

COLORS = np.array([[0, 0, 0], [244, 67, 54], [233, 30, 99], [156, 39, 176], [103, 58, 183], [100, 30, 60],
                   [63, 81, 181], [33, 150, 243], [3, 169, 244], [0, 188, 212], [20, 55, 200],
                   [0, 150, 136], [76, 175, 80], [139, 195, 74], [205, 220, 57], [70, 25, 100],
                   [255, 235, 59], [255, 193, 7], [255, 152, 0], [255, 87, 34], [90, 155, 50],
                   [121, 85, 72], [158, 158, 158], [96, 125, 139], [15, 67, 34], [98, 55, 20],
                   [21, 82, 172], [58, 128, 255], [196, 125, 39], [75, 27, 134], [90, 125, 120],
                   [121, 82, 7], [158, 58, 8], [96, 25, 9], [115, 7, 234], [8, 155, 220],
                   [221, 25, 72], [188, 58, 158], [56, 175, 19], [215, 67, 64], [198, 75, 20],
                   [62, 185, 22], [108, 70, 58], [160, 225, 39], [95, 60, 144], [78, 155, 120],
                   [101, 25, 142], [48, 198, 28], [96, 225, 200], [150, 167, 134], [18, 185, 90],
                   [21, 145, 172], [98, 68, 78], [196, 105, 19], [215, 67, 84], [130, 115, 170],
                   [255, 0, 255], [255, 255, 0], [196, 185, 10], [95, 167, 234], [18, 25, 190],
                   [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [155, 0, 0],
                   [0, 155, 0], [0, 0, 155], [46, 22, 130], [255, 0, 155], [155, 0, 255],
                   [255, 155, 0], [155, 255, 0], [0, 155, 255], [0, 255, 155], [18, 5, 40],
                   [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244], [128, 25, 70]], dtype='uint8')

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

CASTOM_CLASSES = ('first', 'backgrnd')

class Detection(Process):
    """
    Attributes
    ----------
    input_size : int
        Represents the size of the input frame.
    input : Queue
        The queue of input frames for the detection process.
    cfg : dict
        The configuration settings for the rknn-detection process. It include parameters such as 
        confidence thresholds, maximum number of output predictions, etc. (see main.py)
    q_out : Queue
        An instance of the "Queue" class with a maximum size of 3. It is used to store the 
        processed frames and prepared results for display.
    
    Methods
    -------
    permute(net_outputs)
        Permutes the elements in the net_outputs list according to a specific order.
    detect(inputs)
        Detect is the final layer of SSD. Decode location preds, apply non-maximum suppression 
        to location predictions based on conf scores and threshold to a top_k number of output 
        predictions for both confidence score and locations, as the predicted masks.
    prep_display(results)
        This method prepares the results for display. It extracts data from the inference results
        in the form: class_ids, scores, bboxes, masks
    run(None)
        Method runs in an infinite loop. It puts the frame and prepared results into the "q_out" queue.
    
    """
    
    def __init__(self, input, cfg=None):
        super().__init__(group=None, target=None, name=None, args=(), kwargs={}, daemon=True)
        self.input_size = 0
        self.input = input
        self.cfg = cfg
        self.q_out = Queue(maxsize=3)
    
    def run(self):
        while True:
            frame, inputs = self.input.get()
            inputs = self.permute(inputs)
            results = self.detect(inputs)
            self.q_out.put((frame, self.prep_display(results)))
    
    def permute(self, net_outputs):
        '''implementation dependent'''
        pass

    def detect(self, inputs):
        '''implementation dependent'''
        pass

    def prep_display(self, results):
        '''implementation dependent'''
        pass


class ONNXDetection(Detection):
    """This class is a subclass of the Detection class and implements ONNX-based object detection.
    
    Attributes
    ----------
    input_size : int
        The size of the input frame.
    onnx_postprocess : str
        Path to onnx model
    session : onnxruntime.InferenceSession
        Constructs an InferenceSession from a model data (in byte array).
    threshold : int
        Detections with a score under this threshold will not be considered.

    Methods
    -------
    __init__(input, cfg)
        Initializes the ONNXDetection algorithm by creating an InferenceSession
    permute(net_outputs)
        Transposes the arrays in onnx_inputs to have a specific shape and returns 
        the permuted and transposed onnx_inputs.
    detect(onnx_inputs)
        Runs the ONNX session with the given onnx_inputs and returns the outputs of the session.
    prep_display(results)
        Extracts bounding box, score, and class ID from each result. Applies a threshold 
        to filter out low scores.
    
    """

    def __init__(self, input, cfg):
        super().__init__(input, cfg)
        self.input_size = 550
        self.onnx_postprocess = "utils/postprocess_550x550.onnx"
        self.session = onnxruntime.InferenceSession(self.onnx_postprocess,
                                                    None) # providers=OrtSessionOptionsAppendExecutionProvider_RKNPU
        self.threshold = 0.1
    
    def permute(self, net_outputs):
        '''
        Returns
        -------
        post_loc, post_score, post_proto, post_masks
        '''
        onnx_inputs = [net_outputs[0][0], net_outputs[2][0], net_outputs[3], net_outputs[1][0]]
        onnx_inputs[0] = np.transpose(onnx_inputs[0], (2,0,1))
        onnx_inputs[1] = np.transpose(onnx_inputs[1], (2,0,1))
        onnx_inputs[3] = np.transpose(onnx_inputs[3], (2,0,1))
        return onnx_inputs
    
    def detect(self, onnx_inputs):
        '''
        Returns
        -------
        x1y1x2y2_score_class, final_masks
        '''
        results = self.session.run(None, {self.session.get_inputs()[0].name: onnx_inputs[0],
                                          self.session.get_inputs()[1].name: onnx_inputs[1],
                                          self.session.get_inputs()[2].name: onnx_inputs[2],
                                          self.session.get_inputs()[3].name: onnx_inputs[3]})
        return results
    
    def prep_display(self, results):
        def crop(bbox, shape):
            x1 = max(int(bbox[0] * shape[1]), 0)
            y1 = max(int(bbox[1] * shape[0]), 0)
            x2 = max(int(bbox[2] * shape[1]), 0)
            y2 = max(int(bbox[3] * shape[0]), 0)
            return (slice(y1, y2), slice(x1, x2))
        
        bboxes, scores, class_ids, masks = [], [], [], []
        
        for result, mask in zip(results[0][0], results[1]):
            bbox = result[:4].tolist()
            score = result[4]
            class_id = int(result[5])
            
            if self.threshold <= score:
                mask = np.where(mask > 0.5, class_id + 1, 0).astype(np.uint8)
                region = crop(bbox, mask.shape)
                cropped = np.zeros(mask.shape, dtype=np.uint8)
                cropped[region] = mask[region]

                bboxes.append(bbox)
                class_ids.append(class_id)
                scores.append(score)
                masks.append(cropped)
        
        return class_ids, scores, bboxes, masks


class RKNNDetection(Detection):
    """This class represents an implementation of the RKNNDetection algorithm, which is a subclass of the 
    Detection class. It includes methods for initializing the algorithm, permuting the network outputs, 
    performing object detection, and preparing the results for display.
    
    Attributes
    ----------
    input_size : int
        The size of the input frame.
    anchors : list
        A list of anchor boxes used for object detection.

    Methods
    -------
    __init__(input, cfg)
        Initializes the RKNNDetection algorithm by setting the input size and generating the anchor boxes.
    permute(net_outputs)
        Permutes the arrays in net_outputs to have a specific shape.
    detect(onnx_inputs)
        Performs object detection by applying non-maximum suppression.
    prep_display(results)
        Prepares the results for display.
    
    """

    def __init__(self, input, cfg):
        super().__init__(input, cfg)
        self.input_size = 544
        self.anchors = []
        fpn_fm_shape = [math.ceil(self.input_size / stride) for stride in (8, 16, 32, 64, 128)]
        for i, size in enumerate(fpn_fm_shape):
            self.anchors += make_anchors(self.cfg, size, size, self.cfg['scales'][i])
    
    def permute(self, net_outputs):
        '''
        Returns
        -------
        class_p, box_p, coef_p, proto_p
        '''
        class_p, box_p, coef_p, proto_p = net_outputs
        class_p = class_p[0]
        box_p = box_p[0]
        coef_p = coef_p[0]
        class_p = np_softmax(class_p)
        return class_p, box_p, coef_p, proto_p
    
    def detect(self, inputs):
        '''
        Returns
        -------
        class_ids, class_thre, box_thre, coef_thre, proto_p
        '''
        return nms_numpy(*inputs, anchors=self.anchors, cfg=self.cfg)
    
    def prep_display(self, results):
        '''
        Returns
        -------
        ids_p, class_p, box_p, masks
        '''
        return after_nms_numpy(*results, self.input_size, self.input_size, self.cfg)


class PostProcess():
    """Class to handle post-processing of yolact inference results.

    Attributes
    ----------
    detection : Detection
        Detection class object.

    Methods
    -------
    run()
        Starts the detection process.
    get_outputs()
        Retrieves the prepared results from the detection process.
        
    """
    
    def __init__(self, queue, cfg:None, onnx:True):
        """
        Parameters
        ----------
        queue : Queue
            An instance of the "Queue" class with a maximum size of 3, used to store processed frames 
            and prepared results for display.
        cfg : dict
            Configuration settings for the detection process. May include parameters such as 
            confidence thresholds, maximum number of output predictions, etc. Default is None.
        onnx : bool
            Flag indicating whether to use ONNXDetection or RKNNDetection. Default is True.
        """
        if onnx:
            self.detection = ONNXDetection(queue, cfg)
        else:
            self.detection = RKNNDetection(queue, cfg)
    
    def run(self):
        self.detection.start()
    
    def get_outputs(self):
        return self.detection.q_out.get()


def make_anchors(cfg, conv_h, conv_w, scale):
    prior_data = []
    # Iteration order is important (it has to sync up with the convout)
    for j, i in product(range(conv_h), range(conv_w)):
        # + 0.5 because priors are in center
        x = (i + 0.5) / conv_w
        y = (j + 0.5) / conv_h

        for ar in cfg['aspect_ratios']:
            ar_sqrt = sqrt(ar)
            w = scale * ar_sqrt / cfg['img_size']
            h = scale / ar_sqrt / cfg['img_size']

            prior_data.extend([x, y, w, h])

    return prior_data

def np_softmax(x):
    np_max = np.max(x, axis=1)
    sft_max = []
    for idx, pred in enumerate(x):
        e_x = np.exp(pred - np_max[idx])
        sft_max.append(e_x / e_x.sum())
    sft_max = np.array(sft_max)
    return sft_max


class Visualizer():
    
    def __init__(self, onnx=True):
        if onnx:
            self.draw = onnx_draw
        else:
            self.draw = rknn_draw

    def show_results(self, frame, out):
        """
        Show the given frame on the screen with the specified output.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame, _ = self.draw(frame, *out)
        cv2.imshow('Yolact Inference', frame)
        cv2.waitKey(1)
    
    def show_evaluate(self, frame, out, gt_mask, evaluate_results):
        """
        Show the given frame on the screen with the masks and evaluate result.
        """
        accuracy, precision, recall = evaluate_results
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame, mask = self.draw(frame, *out)
        mask = cv2.addWeighted(mask, 0.4, gt_mask, 0.6, gamma=0)
        mask = add_eval_data(mask, np.round(accuracy, 3), np.round(precision, 3), np.round(recall,3))
        cv2.imshow('Yolact Inference', frame)
        cv2.imshow('Masks', mask)
        #cv2.imshow('Ground Truth', gt_mask)
        cv2.waitKey(1)


def onnx_draw(frame, class_ids, scores, bboxes, masks):
    """
    Draw bounding boxes, scores, and masks on a given frame.

    Returns
    -------
    frame : np.ndarray
        The frame with bounding boxes, scores, and masks drawn on it.
    """
    colors = get_colors(len(COCO_CLASSES))
    frame_height, frame_width = frame.shape[0], frame.shape[1]
    # Draw masks
    if len(masks) > 0:
        mask_image = np.zeros(MASK_SHAPE, dtype=np.uint8)
        for mask in masks:
            color_mask = np.array(colors, dtype=np.uint8)[mask]
            filled = np.nonzero(mask)
            mask_image[filled] = color_mask[filled]
        mask_image = cv2.resize(mask_image, (frame_width, frame_height), cv2.INTER_NEAREST)
        cv2.addWeighted(frame, 0.5, mask_image, 0.5, 0.0, frame)

    # Draw boxes
    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        x1, y1 = int(bbox[0] * frame_width), int(bbox[1] * frame_height)
        x2, y2 = int(bbox[2] * frame_width), int(bbox[3] * frame_height)
        color = colors[class_id + 1]
        frame = draw_box(frame, (x1, y1, x2, y2), color, class_id, score)
    return frame, mask_image


def rknn_draw(img_origin, ids_p, class_p, box_p, mask_p, cfg=None, fps=None):
    """
    Generates an image with bounding boxes and labels for detected objects.

    Returns
    -------
    frame : numpy.ndarray
        The image with bounding boxes, masks and labels.
    """
    real_time = False
    if ids_p is None:
        return img_origin

    num_detected = ids_p.shape[0]

    img_fused = img_origin
    masks_semantic = mask_p * (ids_p[:, None, None] + 1)  # expand ids_p' shape for broadcasting
    # The color of the overlap area is different because of the '%' operation.
    masks_semantic = masks_semantic.astype('int').sum(axis=0) % (len(COCO_CLASSES))
    color_masks = COLORS[masks_semantic].astype('uint8')
    img_fused = cv2.addWeighted(color_masks, 0.4, img_origin, 0.6, gamma=0)

    scale = 0.6
    thickness = 1
    font = cv2.FONT_HERSHEY_DUPLEX

    for i in reversed(range(num_detected)):
        color = COLORS[ids_p[i] + 1].tolist()
        img_fused = draw_box(img_fused, box_p[i, :], color, ids_p[i], class_p[i])

    if real_time:
        fps_str = f'fps: {fps:.2f}'
        text_w, text_h = cv2.getTextSize(fps_str, font, scale, thickness)[0]
        # Create a shadow to show the fps more clearly
        img_fused = img_fused.astype(np.float32)
        img_fused[0:text_h + 8, 0:text_w + 8] *= 0.6
        img_fused = img_fused.astype(np.uint8)
        cv2.putText(img_fused, fps_str, (0, text_h + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img_fused, color_masks


def draw_gt(gt_masks):
    masks_semantic = gt_masks.astype('int').sum(axis=0) % (len(COCO_CLASSES))
    colors = get_colors(len(COCO_CLASSES))
    colors = np.array(colors, dtype=np.uint8)
    color_masks = colors[masks_semantic].astype('uint8')
    return color_masks

def draw_box(frame, box, color, class_id, score):
    hide_score = False
    scale = 0.6
    thickness = 1
    font = cv2.FONT_HERSHEY_DUPLEX

    x1, y1, x2, y2 = box
    class_name = COCO_CLASSES[class_id]
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    text_str = f'{class_name}: {score:.2f}' if not hide_score else class_name
    text_w, text_h = cv2.getTextSize(text_str, font, scale, thickness)[0]
    cv2.rectangle(frame, (x1, y1), (x1 + text_w, y1 + text_h + 5), color, -1)
    cv2.putText(frame, text_str, (x1, y1 + 15), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return frame

def get_colors(num):
    colors = [[0, 0, 0]]
    np.random.seed(0)
    for _ in range(num):
        color = np.random.randint(0, 256, [3]).astype(np.uint8)
        colors.append(color.tolist())
    return colors


iou_thres = [x / 100 for x in range(5, 50, 5)]
def evaluate(outputs, ground_truth):
    gt, gt_masks, img_h, img_w = ground_truth
    ap_data = {'box': [[APDataObject() for _ in COCO_CLASSES] for _ in iou_thres],
               'mask': [[APDataObject() for _ in COCO_CLASSES] for _ in iou_thres]}
    
    ids_p, class_p, boxes_p, masks_p = outputs
    ap_obj = ap_data['box'][0][0]
    prep_metrics(ap_data, ids_p, class_p, boxes_p, masks_p, gt, gt_masks, img_h, img_w, iou_thres)
    accuracy, precision, recall = ap_obj.get_accuracy()
    gt_mask = draw_gt(gt_masks)
    return gt_mask, (accuracy, precision, recall)

def add_eval_data(frame, accuracy, precision, recall):
    text_acc = f"accuracy {accuracy}"
    text_pre = f"precision {precision}"
    text_rec = f"recall {recall}"
    cv2.putText(frame, text_acc, (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (255, 125, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, text_pre, (15, 50), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (255, 125, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, text_rec, (15, 75), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (255, 125, 255), 1, cv2.LINE_AA)
    return frame