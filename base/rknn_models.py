from pathlib import Path
from multiprocessing import Process, Queue
import numpy as np
from rknnlite.api import RKNNLite

ROOT = Path(__file__).parent.parent.parent.absolute()
MODELS_PATH = str(ROOT) + "/models/"


class RKNNModelNames():
    NAMES_DICT = {'YOLACT':'yolact.rknn',
                  }
    
    #@classmethod
    def get_model_names(self, model_list):
        path_list = []
        for model in model_list:
            path_list.append(self.NAMES_DICT.get(model))
        return path_list


def get_model_path(model_name):
    try:
        return MODELS_PATH+model_name
    except IndexError:
        return None


class RKNNModelLoader():
    """
    """
    
    def __init__(self):
        self.verbose = False
        self.verbose_file = 'verbose.txt'
        self.async_mode = False
    
    @staticmethod
    def load_rknn_model(core, model):
        model = get_model_path(model)
        rknnlite = RKNNLite()
        print(f"Export rknn model - {model}")
        ret = rknnlite.load_rknn(model)
        if ret != 0:
            print(f'Export {model} model failed!')
            return ret
        print('Init runtime environment')
        ret = rknnlite.init_runtime(async_mode=False,
                                    core_mask = core
                                    )
        if ret != 0:
            print('Init runtime environment failed!')
            return ret
        print(f'{model} is loaded')
        return rknnlite


class Inference(Process):
        def __init__(self, input, output, rknnlite):
            super().__init__(group=None, target=None, name=None, args=(), kwargs={}, daemon=True)
            self.input = input
            self.output = output
            self._rknnlite = rknnlite
        
        def run(self):
            while True:
                frame_list = self.input.get()
                outputs = []
                for frame in frame_list:
                    vector = self._rknnlite.inference(inputs=[frame])
                    outputs.append(vector)
                self.output.put(np.array(outputs))


class YolAct():
    """
    """
    
    def __init__(self, cores,  q_input):
        self._cores = cores
        self.queue = q_input
        self.rknn_model = RKNNModelNames().get_model_names(['YOLACT'])
    
    def load_weights(self):
        self._rknnlite = RKNNModelLoader.load_rknn_model(self._cores, self.rknn_model)
    
    def net_init(self):
        self.load_weights()
        self.inference = Inference(self.queue, Queue(maxsize=3), self._rknnlite)
