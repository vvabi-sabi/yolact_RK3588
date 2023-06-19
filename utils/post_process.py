import time
from itertools import product
import math
from math import sqrt
import cv2
import numpy as np
import onnxruntime


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

INPUT_SIZE = (550, 550)
MASK_SHAPE = (138, 138, 3)

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
        bboxes, scores, class_ids, masks = self.run(onnx_out,
                                                    self.input_size,
                                                    score_th=self.threshold)
        return bboxes, scores, class_ids, masks
        
    
    def transpose_input(self, onnx_inputs):
        onnx_inputs = [onnx_inputs[1][0], onnx_inputs[0][0], onnx_inputs[3], onnx_inputs[2][0]]
        onnx_inputs[0] = np.transpose(onnx_inputs[0], (2,0,1))
        onnx_inputs[1] = np.transpose(onnx_inputs[1], (2,0,1))
        onnx_inputs[3] = np.transpose(onnx_inputs[3], (2,0,1))
        return onnx_inputs
    
    def run(self,results, input_size, score_th):
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
        ids_p, class_p, boxes_p, masks_p = after_nms_numpy(ids_p, class_p, box_p, coef_p, proto_p,
                                                           self.img_h, self.img_w, self.cfg)
        return ids_p, class_p, boxes_p, masks_p
    
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

def box_iou_numpy(box_a, box_b):
    (n, A), B = box_a.shape[:2], box_b.shape[1]
    # add a dimension
    box_a = np.tile(box_a[:, :, None, :], (1, 1, B, 1))
    box_b = np.tile(box_b[:, None, :, :], (1, A, 1, 1))

    max_xy = np.minimum(box_a[..., 2:], box_b[..., 2:])
    min_xy = np.maximum(box_a[..., :2], box_b[..., :2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=100000)
    inter_area = inter[..., 0] * inter[..., 1]

    area_a = (box_a[..., 2] - box_a[..., 0]) * (box_a[..., 3] - box_a[..., 1])
    area_b = (box_b[..., 2] - box_b[..., 0]) * (box_b[..., 3] - box_b[..., 1])

    return inter_area / (area_a + area_b - inter_area)

def fast_nms_numpy(box_thre, coef_thre, class_thre, cfg):
    # descending sort
    idx = np.argsort(-class_thre, axis=1)
    class_thre = np.sort(class_thre, axis=1)[:, ::-1]
    idx = idx[:, :cfg['top_k']]
    class_thre = class_thre[:, :cfg['top_k']]
    num_classes, num_dets = idx.shape
    box_thre = box_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)  # [80, 64, 4]
    coef_thre = coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]
    iou = box_iou_numpy(box_thre, box_thre)
    iou = np.triu(iou, k=1)
    iou_max = np.max(iou, axis=1)
    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= cfg['nms_iou_thre'])
    # Assign each kept detection to its corresponding class
    class_ids = np.tile(np.arange(num_classes)[:, None], (1, keep.shape[1]))
    class_ids, box_nms, coef_nms, class_nms = class_ids[keep], box_thre[keep], coef_thre[keep], class_thre[keep]
    # Only keep the top cfg.max_num_detections highest scores across all classes
    idx = np.argsort(-class_nms, axis=0)
    class_nms = np.sort(class_nms, axis=0)[::-1]

    idx = idx[:cfg['max_detections']]
    class_nms = class_nms[:cfg['max_detections']]

    class_ids = class_ids[idx]
    box_nms = box_nms[idx]
    coef_nms = coef_nms[idx]

    return box_nms, coef_nms, class_ids, class_nms

def nms_numpy(class_pred, box_pred, coef_pred, proto_out, anchors, cfg):
    class_p = class_pred.squeeze()  # [19248, 81]
    box_p = box_pred.squeeze()  # [19248, 4]
    coef_p = coef_pred.squeeze()  # [19248, 32]
    proto_p = proto_out.squeeze()  # [138, 138, 32]
    anchors = np.array(anchors).reshape(-1, 4)

    class_p = class_p.transpose(1, 0)
    # exclude the background class
    class_p = class_p[1:, :]
    # get the max score class of 19248 predicted boxes
    class_p_max = np.max(class_p, axis=0)  # [19248]
    # filter predicted boxes according the class score
    keep = (class_p_max > cfg['nms_score_thre'])
    class_thre = class_p[:, keep]
    box_thre, anchor_thre, coef_thre = box_p[keep, :], anchors[keep, :], coef_p[keep, :]
    # decode boxes
    box_thre = np.concatenate((anchor_thre[:, :2] + box_thre[:, :2] * 0.1 * anchor_thre[:, 2:],
                               anchor_thre[:, 2:] * np.exp(box_thre[:, 2:] * 0.2)), axis=1)
    box_thre[:, :2] -= box_thre[:, 2:] / 2
    box_thre[:, 2:] += box_thre[:, :2]

    print('box_thre', box_thre.shape)
    if class_thre.shape[1] == 0:
        return None, None, None, None, None
    else:
        assert not cfg['traditional_nms'], 'Traditional nms is not supported with numpy.'
        box_thre, coef_thre, class_ids, class_thre = fast_nms_numpy(box_thre, coef_thre, class_thre, cfg)
        return class_ids, class_thre, box_thre, coef_thre, proto_p

def sanitize_coordinates_numpy(_x1, _x2, img_size, padding=0):
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size

    x1 = np.minimum(_x1, _x2)
    x2 = np.maximum(_x1, _x2)
    x1 = np.clip(x1 - padding, a_min=0, a_max=1000000)
    x2 = np.clip(x2 + padding, a_min=0, a_max=img_size)

    return x1, x2

def crop_numpy(masks, boxes, padding=1):
    h, w, n = masks.shape
    x1, x2 = sanitize_coordinates_numpy(boxes[:, 0], boxes[:, 2], w, padding)
    y1, y2 = sanitize_coordinates_numpy(boxes[:, 1], boxes[:, 3], h, padding)

    rows = np.tile(np.arange(w)[None, :, None], (h, 1, n))
    cols = np.tile(np.arange(h)[:, None, None], (1, w, n))

    masks_left = rows >= (x1.reshape(1, 1, -1))
    masks_right = rows < (x2.reshape(1, 1, -1))
    masks_up = cols >= (y1.reshape(1, 1, -1))
    masks_down = cols < (y2.reshape(1, 1, -1))

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask

def after_nms_numpy(ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, cfg=None):
    def np_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    if ids_p is None:
        return None, None, None, None

    print(cfg.visual_thre)
    if cfg and cfg.visual_thre > 0:
        keep = class_p >= cfg.visual_thre
        if not keep.any():
            return None, None, None, None

        print(len(keep))
        ids_p = ids_p[keep]
        class_p = class_p[keep]
        box_p = box_p[keep]
        coef_p = coef_p[keep]

    assert not cfg.save_lincomb, 'save_lincomb is not supported in onnx mode.'

    masks = np_sigmoid(np.matmul(proto_p, coef_p.T))

    if not cfg or not cfg.no_crop:  # Crop masks by box_p
        masks = crop_numpy(masks, box_p)

    ori_size = max(img_h, img_w)
    masks = cv2.resize(masks, (ori_size, ori_size), interpolation=cv2.INTER_LINEAR)

    if masks.ndim == 2:
        masks = masks[:, :, None]

    masks = np.transpose(masks, (2, 0, 1))
    masks = masks > 0.5  # Binarize the masks because of interpolation.
    masks = masks[:, 0: img_h, :] if img_h < img_w else masks[:, :, 0: img_w]

    box_p *= ori_size
    box_p = box_p.astype('int32')

    return ids_p, class_p, box_p, masks


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
    for i in range(num):
        color = np.random.randint(0, 256, [3]).astype(np.uint8)
        colors.append(color.tolist())
    return colors

def draw(frame, elapsed_time, bboxes, scores, class_ids, masks):
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