import time
from pathlib import Path
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
    
    def __init__(self) -> None:
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
        bboxes, scores, class_ids, masks = run_inference(onnx_out,
                                                         self.input_size,
                                                         score_th=self.threshold)
        return bboxes, scores, class_ids, masks
        
    
    def transpose_input(self, onnx_inputs):
        onnx_inputs = [onnx_inputs[1][0], onnx_inputs[0][0], onnx_inputs[3], onnx_inputs[2][0]]
        onnx_inputs[0] = np.transpose(onnx_inputs[0], (2,0,1))
        onnx_inputs[1] = np.transpose(onnx_inputs[1], (2,0,1))
        onnx_inputs[3] = np.transpose(onnx_inputs[3], (2,0,1))
        return onnx_inputs


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