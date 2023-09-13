import cv2
import numpy as np

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
    box_thre = box_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)
    coef_thre = coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)
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

    if class_thre.shape[1] == 0:
        return None, None, None, None, None
    else:
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

    if cfg and cfg['visual_thre'] > 0:
        keep = class_p >= cfg['visual_thre']
        if not keep.any():
            return None, None, None, None

        ids_p = ids_p[keep]
        class_p = class_p[keep]
        box_p = box_p[keep]
        coef_p = coef_p[keep]

    masks = np_sigmoid(np.matmul(proto_p, coef_p.T))
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


def mask_iou(mask1, mask2):
    """
    Inputs inputs are matricies of size _ x N. Output is size _1 x _2.
    Note: if iscrowd is True, then mask2 should be the crowd.
    """
    intersection = np.matmul(mask1, mask2.T)
    area1 = np.sum(mask1, axis=1).reshape(1, -1)
    area2 = np.sum(mask2, axis=1).reshape(1, -1)
    union = (area1.T + area2) - intersection
    ret = intersection / union

    return ret