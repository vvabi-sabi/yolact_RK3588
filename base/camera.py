import cv2
from multiprocessing import Process


class Camera(Process):
    """
    """
    def __init__(self, source: int, queue):
        super().__init__(group=None, target=None, name=None, args=(), kwargs={}, daemon=True)
        self._queue = queue
        self.source = source

    @property
    def frames(self):
        cap = cv2.VideoCapture(self.source)
        if(not cap.isOpened()):
            print("Bad source")
            raise SystemExit
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Camera stopped!")
                    raise SystemExit
                yield frame
            cap.release()
        except Exception as e:
            print("Stop recording loop. Exception {}".format(e))
    
    def get_frame(self):
        return next(self.frames)
    
    def run(self):
        net_size = (544, 544)
        for raw_frame in self.frames:
            #frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(raw_frame.copy(), net_size)
            self._queue.put((frame))