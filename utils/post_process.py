import time
from itertools import product
import math
from math import sqrt
import cv2
import numpy as np
import onnxruntime
from box_utils import nms_numpy, after_nms_numpy


postprocess_type = 'onnx' # 'rknn'
INPUT_SIZE = ((550,550) if postprocess_type =='onnx' else (544,544))
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


class OnnxPostProcess():
    
    def __init__(self):
        self.onnx_postprocess = "postprocess_550x550.onnx"
        self.input_size = INPUT_SIZE
        self.threshold = 0.1
        self.session = onnxruntime.InferenceSession(self.onnx_postprocess,
                                                    None)
    
    def process(self, onnx_inputs):
        onnx_inputs = self.transpose_input(onnx_inputs)
        onnx_out = self.session.run(None, {self.session.get_inputs()[0].name: onnx_inputs[0],
                                           self.session.get_inputs()[1].name: onnx_inputs[1],
                                           self.session.get_inputs()[2].name: onnx_inputs[2],
                                           self.session.get_inputs()[3].name: onnx_inputs[3]})
        return self.run(onnx_out, score_th=self.threshold)
        
    
    def transpose_input(self, onnx_inputs):
        onnx_inputs = [onnx_inputs[1][0], onnx_inputs[0][0], onnx_inputs[3], onnx_inputs[2][0]]
        onnx_inputs[0] = np.transpose(onnx_inputs[0], (2,0,1))
        onnx_inputs[1] = np.transpose(onnx_inputs[1], (2,0,1))
        onnx_inputs[3] = np.transpose(onnx_inputs[3], (2,0,1))
        return onnx_inputs
    
    def run(self,results, score_th):
        # Pre process: Creates 4-dimensional blob from image
        def crop(bbox, shape):
            x1 = int(max(bbox[0] * shape[1], 0))
            y1 = int(max(bbox[1] * shape[0], 0))
            x2 = int(max(bbox[2] * shape[1], 0))
            y2 = int(max(bbox[3] * shape[0], 0))
            return (slice(y1, y2), slice(x1, x2))

        # Post process
        bboxes, scores, class_ids, masks = [], [], [], []
        for result, mask in zip(results[0][0], results[1]):
            bbox = result[:4].tolist()
            score = result[4]
            class_id = int(result[5])

            if score_th > score:
                continue

            # Add 1 to class_id to distinguish it from the background 0
            mask = np.where(mask > 0.5, class_id + 1, 0).astype(np.uint8)
            region = crop(bbox, mask.shape)
            cropped = np.zeros(mask.shape, dtype=np.uint8)
            cropped[region] = mask[region]

            bboxes.append(bbox)
            class_ids.append(class_id)
            scores.append(score)
            masks.append(cropped)

        return bboxes, scores, class_ids, masks
        


class RknnPostProcess():
    
    def __init__(self):
        self.img_h = INPUT_SIZE[0]
        self.img_w = INPUT_SIZE[1]
        self.cfg = {'weight':'weights/best_30.5_res101_coco_392000.pth',
                    'image': 'test_544.jpg',
                    'video' : None,
                    'img_size' : 544,
                    'traditional_nms' : False,
                    'hide_mask' : False,
                    'hide_bbox' : False,
                    'hide_score' : False,
                    'cutout' : False,
                    'save_lincomb' : False,
                    'no_crop' : False,
                    'real_time' : False,
                    'scales' : [24, 48, 96, 192, 384],
                    'top_k' : 200,
                    'max_detections' : 100,
                    'nms_score_thre' : 0.5,
                    'nms_iou_thre' : 0.5,
                    'visual_thre' : 0.3,
                  }
        self.anchors = []
        fpn_fm_shape = [math.ceil(INPUT_SIZE / stride) for stride in (8, 16, 32, 64, 128)]
        for i, size in enumerate(fpn_fm_shape):
            self.anchors += make_anchors(self.cfg, size, size, self.cfg['scales'][i])
    
    def process(self, rknn_outputs):
        class_p, box_p, coef_p, proto_p = self.get_outputs(rknn_outputs)
        ids_p, class_p, box_p, coef_p, proto_p = nms_numpy(class_p,
                                                           box_p,
                                                           coef_p,
                                                           proto_p,
                                                           self.anchors,
                                                           self.cfg)
        return after_nms_numpy(ids_p, class_p, box_p, coef_p, proto_p,
                               self.img_h, self.img_w, self.cfg)
    
    def get_outputs(self, rknn_outputs):
        class_p, box_p, coef_p, proto_p = rknn_outputs
        class_p = class_p[0]
        box_p = box_p[0]
        coef_p = coef_p[0]
        class_p = np_softmax(class_p)
        return class_p, box_p, coef_p, proto_p



class Visualizer():
    
    @classmethod
    def onnx_draw(frame, elapsed_time, bboxes, scores, class_ids, masks):
        colors = get_colors(len(COCO_CLASSES))
        frame_height, frame_width = frame.shape[0], frame.shape[1]
        cv2.putText(frame,
                    "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

        # Draw
        if len(masks) > 0:
            mask_image = np.zeros(MASK_SHAPE, dtype=np.uint8)
            for mask in masks:
                color_mask = np.array(colors, dtype=np.uint8)[mask]
                filled = np.nonzero(mask)
                mask_image[filled] = color_mask[filled]
            mask_image = cv2.resize(mask_image, (frame_width, frame_height), cv2.INTER_NEAREST)
            cv2.addWeighted(frame, 0.5, mask_image, 0.5, 0.0, frame)

        for bbox, score, class_id, mask in zip(bboxes, scores, class_ids, masks):
            x1, y1 = int(bbox[0] * frame_width), int(bbox[1] * frame_height)
            x2, y2 = int(bbox[2] * frame_width), int(bbox[3] * frame_height)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, '%s:%.2f' % (COCO_CLASSES[class_id], score),
                    (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
    
    @classmethod
    def rknn_draw(ids_p, class_p, box_p, mask_p, img_origin, cfg, img_name=None, fps=None):
        if ids_p is None:
            return img_origin

        num_detected = ids_p.shape[0]

        img_fused = img_origin
        if not cfg.hide_mask:
            masks_semantic = mask_p * (ids_p[:, None, None] + 1)  # expand ids_p' shape for broadcasting
            # The color of the overlap area is different because of the '%' operation.
            masks_semantic = masks_semantic.astype('int').sum(axis=0) % (cfg.num_classes - 1)
            color_masks = COLORS[masks_semantic].astype('uint8')
            img_fused = cv2.addWeighted(color_masks, 0.4, img_origin, 0.6, gamma=0)

            if cfg.cutout:
                total_obj = (masks_semantic != 0)[:, :, None].repeat(3, 2)
                total_obj = total_obj * img_origin
                new_mask = ((masks_semantic == 0) * 255)[:, :, None].repeat(3, 2)
                img_matting = (total_obj + new_mask).astype('uint8')
                cv2.imwrite(f'results/images/{img_name}_total_obj.jpg', img_matting)

                for i in range(num_detected):
                    one_obj = (mask_p[i])[:, :, None].repeat(3, 2)
                    one_obj = one_obj * img_origin
                    new_mask = ((mask_p[i] == 0) * 255)[:, :, None].repeat(3, 2)
                    x1, y1, x2, y2 = box_p[i, :]
                    img_matting = (one_obj + new_mask)[y1:y2, x1:x2, :]
                    cv2.imwrite(f'results/images/{img_name}_{i}.jpg', img_matting)
        scale = 0.6
        thickness = 1
        font = cv2.FONT_HERSHEY_DUPLEX

        if not cfg.hide_bbox:
            for i in reversed(range(num_detected)):
                x1, y1, x2, y2 = box_p[i, :]

                color = COLORS[ids_p[i] + 1].tolist()
                cv2.rectangle(img_fused, (x1, y1), (x2, y2), color, thickness)

                class_name = cfg.class_names[ids_p[i]]
                text_str = f'{class_name}: {class_p[i]:.2f}' if not cfg.hide_score else class_name

                text_w, text_h = cv2.getTextSize(text_str, font, scale, thickness)[0]
                cv2.rectangle(img_fused, (x1, y1), (x1 + text_w, y1 + text_h + 5), color, -1)
                cv2.putText(img_fused, text_str, (x1, y1 + 15), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        if cfg.real_time:
            fps_str = f'fps: {fps:.2f}'
            text_w, text_h = cv2.getTextSize(fps_str, font, scale, thickness)[0]
            # Create a shadow to show the fps more clearly
            img_fused = img_fused.astype(np.float32)
            img_fused[0:text_h + 8, 0:text_w + 8] *= 0.6
            img_fused = img_fused.astype(np.uint8)
            cv2.putText(img_fused, fps_str, (0, text_h + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return img_fused


def make_anchors(cfg, conv_h, conv_w, scale):
    prior_data = []
    # Iteration order is important (it has to sync up with the convout)
    for j, i in product(range(conv_h), range(conv_w)):
        # + 0.5 because priors are in center
        x = (i + 0.5) / conv_w
        y = (j + 0.5) / conv_h

        for ar in cfg.aspect_ratios:
            ar = sqrt(ar)
            w = scale * ar / cfg['img_size']
            h = scale / ar / cfg['img_size']

            prior_data += [x, y, w, h]

    return prior_data

def np_softmax(x):
    np_max = np.max(x, axis=1)
    sft_max = []
    for idx, pred in enumerate(x):
        e_x = np.exp(pred - np_max[idx])
        sft_max.append(e_x / e_x.sum())
    sft_max = np.array(sft_max)
    return sft_max




def post_yolact(outputs, frame):
    onnx_postprocess = "postprocess_550x550.onnx"
    input_size = INPUT_SIZE
    threshold = 0.1
    session = onnxruntime.InferenceSession(onnx_postprocess,
                                           None)
    onnx_inputs = get_onnx_inputs(outputs)
    start_time = time.time()
    onnx_out = session.run(None, {session.get_inputs()[0].name: onnx_inputs[0],
                                  session.get_inputs()[1].name: onnx_inputs[1],
                                  session.get_inputs()[2].name: onnx_inputs[2],
                                  session.get_inputs()[3].name: onnx_inputs[3]})
    bboxes, scores, class_ids, masks = run_inference(onnx_out,
                                                     input_size,
                                                     score_th=threshold)
    elapsed_time = time.time() - start_time
    draw(frame, elapsed_time, bboxes, scores, class_ids, masks)



#__YOLACT__
def get_onnx_inputs(rknn_outputs):
    onnx_inputs = [rknn_outputs[1][0], rknn_outputs[0][0], rknn_outputs[3], rknn_outputs[2 ][0]]
    onnx_inputs[0] = np.transpose(onnx_inputs[0], (2,0,1))
    onnx_inputs[1] = np.transpose(onnx_inputs[1], (2,0,1))
    onnx_inputs[3] = np.transpose(onnx_inputs[3], (2,0,1))
    return onnx_inputs

def run_inference(results, input_size, score_th):
    # Pre process: Creates 4-dimensional blob from image
    size = (input_size[1], input_size[0])
    #input_image = cv.dnn.blobFromImage(image, size=size, swapRB=True)

    #results = onnx_session.run(output_names, {input_name: input_image})

    def crop(bbox, shape):
        x1 = int(max(bbox[0] * shape[1], 0))
        y1 = int(max(bbox[1] * shape[0], 0))
        x2 = int(max(bbox[2] * shape[1], 0))
        y2 = int(max(bbox[3] * shape[0], 0))
        return (slice(y1, y2), slice(x1, x2))

    # Post process
    bboxes, scores, class_ids, masks = [], [], [], []
    for result, mask in zip(results[0][0], results[1]):
        bbox = result[:4].tolist()
        score = result[4]
        class_id = int(result[5])

        if score_th > score:
            continue

        # Add 1 to class_id to distinguish it from the background 0
        mask = np.where(mask > 0.5, class_id + 1, 0).astype(np.uint8)
        region = crop(bbox, mask.shape)
        cropped = np.zeros(mask.shape, dtype=np.uint8)
        cropped[region] = mask[region]

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)
        masks.append(cropped)

    return bboxes, scores, class_ids, masks

def get_colors(num):
    colors = [[0, 0, 0]]
    np.random.seed(0)
    for _ in range(num):
        color = np.random.randint(0, 256, [3]).astype(np.uint8)
        colors.append(color.tolist())
    return colors