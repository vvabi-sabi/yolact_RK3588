import cv2
from multiprocessing import Process


class Camera(Process):
    """
    """
    def __init__(self, source: int, queue, onnx=True):
        super().__init__(group=None, target=None, name=None, args=(), kwargs={}, daemon=True)
        INPUT_SIZE = (550 if onnx else 544)
        self.net_size = (INPUT_SIZE, INPUT_SIZE) 
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
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # not always necessary
                yield frame
            cap.release()
        except Exception as e:
            print(f"Stop recording loop. Exception {e}")
    
    def get_frame(self):
        return next(self.frames)
    
    def resize_frame(self, frame, net_size):
        frame_size = frame.shape[:2]
        if any(map(lambda x,y: x < y, frame_size, net_size)):
            return cv2.resize(frame, net_size, interpolation = cv2.INTER_CUBIC)
        else:
            return cv2.resize(frame, net_size, interpolation = cv2.INTER_AREA)

    def crop_frame(self, frame, net_size):
        net_size = net_size[0]
        hc, wc = frame.shape[0]/2, frame.shape[1]/2
        h0, w0 = int(hc-net_size/2), int(wc-net_size/2)
        assert (h0 >= 0 and w0 >= 0), 'The image size is not suitable to crop. Try Camera.resize_frame()'
        return frame[h0:(h0+net_size), w0:(w0+net_size)]

    def run(self):
        for raw_frame in self.frames:
            #frame = self.resize_frame(raw_frame, self.net_size) #cv2.resize(raw_frame.copy(), self.net_size)
            frame = self.crop_frame(raw_frame, self.net_size)
            self._queue.put((frame))