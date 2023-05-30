from rknnlite.api import RKNNLite

from rknn_models import YolAct


class NeuroModule():
    
    _PROCESSES_NUMBER = 3

    def __init__(self, cores_list, q_input):
        self.net = YolAct(cores_list, q_input)

    def get_output(self):
        return self.net.inference.q_out.get()

    def run_inference(self):
        inf_process = self.net.inference
        inf_process.start()


class RK3588():
    """
    """
    _CORES = [RKNNLite.NPU_CORE_0,
              RKNNLite.NPU_CORE_1,
              RKNNLite.NPU_CORE_2,
              RKNNLite.NPU_CORE_AUTO,
              RKNNLite.NPU_CORE_0_1,
              RKNNLite.NPU_CORE_0_1_2
              ]

    def __init__(self, camera):
        self._camera = camera
        self._neuro = NeuroModule(self._CORES[3],
                                  self._camera._queue)
    def get_neuro_outputs(self):
        return self._neuro.get_output()