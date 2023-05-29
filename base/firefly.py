from rknnlite.api import RKNNLite

from rknn_models import YolAct


class NeuroModule():
    
    _PROCESSES_NUMBER = 3

    def __init__(self, cores_list, q_input):
        self.model = YolAct(cores_list)
        self.nets = self.model.net_init(q_input)

    def forward(self):
        return self.net.inference.output.get()

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

    def __init__(self, camera, models_list):
        self._camera = camera
        self._neuro = NeuroModule(self._CORES[3],
                                  self._camera._queue) # ['Model_1', 'Model_2'])

    def get_neuro_outputs(self):
        outputs = self._neuro.forward()
        return outputs