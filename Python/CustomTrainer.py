from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper, build_detection_test_loader

from LossEvalHook import LossEvalHook

class CustomTrainer(DefaultTrainer):
    def __init__(self, cfg, val_data, val_period):
        self.val_data = val_data
        self.val_period = val_period
        self.cfg = cfg
        super().__init__(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, ("bbox",), False, cfg.OUTPUT_DIR)
                     
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg,
            self.val_period,
            self.model,
            self.scheduler,
            build_detection_test_loader(
                self.cfg,
                self.val_data,
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks