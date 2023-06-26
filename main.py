from multiprocessing import Queue

from utils import PostProcess, Visualizer
from base import Camera, RK3588


postprocess_cfg = {'weight':'weights/best_30.5_res101_coco_392000.pth',
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
                    'aspect_ratios': [1, 0.5, 2],
                    'top_k' : 200,
                    'max_detections' : 100,
                    'nms_score_thre' : 0.5,
                    'nms_iou_thre' : 0.5,
                    'visual_thre' : 0.3,
                    }


def run(device, visualizer, post_process):
    device._camera.start()
    device._neuro.run_inference()
    if post_process is not None:
        post_process.run()
        while True:
            frame, outputs = post_process.get_outputs()
            visualizer.show_frame(frame, outputs)

def main(source):
    """
    """
    queue_size = 5
    q_pre = Queue(maxsize=queue_size)
    q_post = Queue(maxsize=queue_size)
    model = 'YOLACT' # 'YOLACT_EDGE'
    camera = Camera(source=source,
                    queue=q_pre)
    device = RK3588(camera)
    post_processes = PostProcess(queue=device._neuro.net.inference.q_out,
                                 cfg=postprocess_cfg,
                                 onnx=False) #q_post)
    visualizer = Visualizer(onnx=False)
    try:
        run(device, visualizer, post_processes)
    except Exception as e:
        print("Main exception: {}".format(e))
        exit()


if __name__ == "__main__":
    camera_source = 11 # 'path/to/video.mp3'
    main(camera_source)
