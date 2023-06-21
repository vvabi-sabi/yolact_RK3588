from multiprocessing import Queue

from utils import RknnPostProcess, Visualizer
from base import Camera, RK3588


def run(device, visualizer, post_process):
    device._camera.start()
    device._neuro.run_inference()
    if post_process is not None:
        post_process.run()
        while True:
            frame, outputs = device.get_neuro_outputs()
            out = post_process(outputs)
            visualizer.show_frame(frame, out)

def main(source):
    """
    """
    queue_size = 5
    q_pre = Queue(maxsize=queue_size)
    q_post = Queue(maxsize=queue_size)
    model = 'YOLACT' # 'YOLACT_EDGE'
    camera = Camera(source=source,
                    queue=q_pre,)
    device = RK3588(camera, model)
    post_processes = RknnPostProcess(queue=q_post)
    visualizer = Visualizer(onnx=False)
    try:
        run(device, visualizer, post_processes)
    except Exception as e:
        print("Main exception: {}".format(e))
        exit()


if __name__ == "__main__":
    camera_source = 11 # 'path/to/video.mp3'
    main(camera_source)
