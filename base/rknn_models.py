from pathlib import Path
from multiprocessing import Process, Queue
from rknnlite.api import RKNNLite

ROOT = Path(__file__).parent.parent.parent.absolute()
MODELS_PATH = str(ROOT) + "/models/"


RKNNModelNames = {'YOLACT':'yolact.rknn'}

def get_model_names(model_list):
    path_list = []
    for model in model_list:
        path_list.append(RKNNModelNames.get(model))
    return path_list

def get_model_path(model_name):
    return MODELS_PATH+model_name



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
    
    def __init__(self, input, rknnlite):
        super().__init__(group=None, target=None, name=None, args=(), kwargs={}, daemon=True)
        self.input = input
        self.q_out = Queue(maxsize=3)
        self._rknnlite = rknnlite
    
    def run(self):
        while True:
            frame = self.input.get()
            self.q_out.put(self._rknnlite.inference(inputs=[frame]))


class YolAct():
    """
    """
    
    def __init__(self, cores, q_input):
        self._model_name = RKNNModelNames.get_model_names(['YOLACT'])
        self._rknnlite = RKNNModelLoader.load_rknn_model(cores, self._model_name)
        self.inference = Inference(q_input, self._rknnlite)