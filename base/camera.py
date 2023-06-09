import cv2
from multiprocessing import Process


class Camera(Process):
    """This class represents a camera process that captures frames from a video source 
    and performs various operations on the frames (preprocess).

    Attributes
    ----------
    net_size : tuple
        The size of the neural network input.
    queue : Queue
        The queue to put the processed frames into.
    source : int, str
        The video source (also path to file.mp4) to capture frames from.
    frames : generator
        A generator object that yields frames from the video capture.

    Methods
    -------
    get_frame(None)
        Returns the next frame from the frames generator.
    resize_frame(frame, net_size)
        Resizes the given frame using OpenCV's resize function.
    crop_frame(frame, net_size)
        Crops the given frame based on net_size.
    run(None) 
        Iterates over the frames generator, processes each frame, and puts it into the queue.

    """

    def __init__(self, source: int, queue, onnx=True):
        """
        Parameters
        ----------
        source : int, str
            The video source.
        queue : Queue
            The queue in which processed frames are placed. Then these frames will be fed 
            to the input of the neural network.
        onnx : bool, optional
            Whether to use ONNX model for postprocessing. Defaults to True.
        """
        super().__init__(group=None, target=None, name=None, args=(), kwargs={}, daemon=True)
        INPUT_SIZE = (550 if onnx else 544)
        self.net_size = (INPUT_SIZE, INPUT_SIZE) 
        self._queue = queue
        self.source = source

    @property
    def frames(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise SystemExit("Bad source")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    raise SystemExit("Camera stopped!")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame
        except Exception as e:
            print(f"Stop recording loop. Exception {e}")
        finally:
            cap.release()
    
    def get_frame(self):
        """It yields the frame, making it available for further processing outside the function.
        """
        return next(self.frames)
    
    def resize_frame(self, frame, net_size):
        frame_size = frame.shape[:2]
        interpolation = cv2.INTER_CUBIC if any(x < y for x, y in zip(frame_size, net_size)) else cv2.INTER_AREA
        return cv2.resize(frame, net_size, interpolation=interpolation)

    def crop_frame(self, frame, net_size):
        net_size = net_size[0]
        hc, wc = frame.shape[0] // 2, frame.shape[1] // 2
        h0, w0 = hc - (net_size // 2), wc - (net_size // 2)
        assert (h0 >= 0 and w0 >= 0), 'The image size is not suitable to crop. Try Camera.resize_frame()'
        return frame[h0:h0+net_size, w0:w0+net_size]

    def run(self):
        '''When processing a raw frame, there are two methods to choose from:
        resize_frame or crop_frame.
        '''
        for raw_frame in self.frames:
            frame = self.resize_frame(raw_frame, self.net_size) #cv2.resize(raw_frame.copy(), self.net_size)
            #frame = self.crop_frame(raw_frame, self.net_size)
            if (not self._queue.empty() and type(self.source) == int):
                continue
            self._queue.put((frame))
