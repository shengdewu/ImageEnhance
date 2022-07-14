from .inference_gt import InferenceNoneGt
import logging


class InferenceNoneGtOld(InferenceNoneGt):
    def __init__(self, cfg, tif=True):
        super(InferenceNoneGt, self).__init__(cfg, tif)
        self.log_name = cfg.OUTPUT_LOG_NAME
        return

    def resume_or_load(self):
        model_state_dict, _ = self.checkpointer.resume_or_load(self.model_path, resume=False)
        model_state_dict['g_model'] = model_state_dict['model']
        self.model.load_state_dict(model_state_dict)
        logging.getLogger(self.log_name).info('load model from {}'.format(self.model_path))
        return
