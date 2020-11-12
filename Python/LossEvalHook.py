import json
import math
import os
import time
import torch
import datetime
import logging
import numpy as np

from detectron2.engine.hooks import HookBase
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm

class LossEvalHook(HookBase):
    def __init__(self, cfg, val_period, model, scheduler, data_loader):
        self._threshold = 0.005 # this threshold has to be surpassed for the mean_loss
        self._best_AP = 0
        self._saved_model_name = None
        self._scheduler = scheduler

        self._cfg = cfg
        self._period = val_period
        self._model = model
        self._data_loader = data_loader
        self._logger = logging.getLogger("detectron2")

        self._is_initialized = False

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        evaluated = 0
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                if idx >= evaluated + 30: # log every 30
                    evaluated = idx
                    self._logger.info("Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(idx + 1, total, seconds_per_img, str(eta)))
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        get_event_storage().put_scalar('validation_loss', mean_loss, smoothing_hint=False)
        comm.synchronize()

        return losses, mean_loss
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
    def _initialize(self):
        found_models = []
        # if there are earlier models - get logged variables from metrics
        for files in os.listdir(self._cfg.OUTPUT_DIR):
            if "model" in files:
                found_models.append(files)
            
        if len(found_models) > 0:
            self._saved_model_name = found_models[-1]

            lines = []
            with open(os.path.join(self._cfg.OUTPUT_DIR, "metrics.json"), 'r') as f:
                for line in f:
                    json_line = json.loads(line)
                    lines.append(json_line)
                    if json_line["iteration"] == self.trainer.iter-1:
                        break
            # to remove any metrics that have been generated from future iterations
            with open(os.path.join(self._cfg.OUTPUT_DIR, "metrics.json"), "w") as f:
                for line in lines:
                    f.write(json.dumps(line))
                    f.write("\n")

            best_model_iteration = 0
            for x in lines:
                if "bbox/AP" in x:
                    if self._best_AP < x["bbox/AP"]:
                        self._best_AP = x["bbox/AP"]
                        best_model_iteration = x["iteration"]

            self._saved_model_name = "model_" + ("%07.0f" % best_model_iteration) + ".pth"
            self._logger.info("found best model: " + self._saved_model_name)
            self._logger.info("with best AP: " + str(self._best_AP))

    def after_step(self):
        if not self._is_initialized:
            self._initialize()
            self._is_initialized = True
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            losses, mean_loss = self._do_loss_eval()

            if is_final:
                model_name = "model_final.pth"
                prev_model_name = "model_0019799.pth"
            else:
                model_name = "model_" + ("%07.0f" % self.trainer.iter) + ".pth"
                prev_model_name = "model_" + ("%07.0f" % (self.trainer.iter-self._period)) + ".pth"

            current_AP = self.trainer.storage._latest_scalars["bbox/AP"]
            if isinstance(current_AP, tuple): # on windows latest_scalars will be a tuple, on linux not
                current_AP = current_AP[0]
            self._logger.info("current AP: " + str(current_AP) + ", best AP: " + str(self._best_AP))
            if self._saved_model_name is None:
                self._saved_model_name = model_name
                self._best_AP = current_AP
            elif current_AP > self._best_AP:
                # remove old model and save new model
                self._logger.info("found model with better AP, saving model : " + model_name)
                self._logger.info("removing previous best model: " + self._saved_model_name)
                os.remove(os.path.join(self._cfg.OUTPUT_DIR, self._saved_model_name))

                if self._saved_model_name != prev_model_name:
                    self._logger.info("remove previous model : " + prev_model_name)
                    os.remove(os.path.join(self._cfg.OUTPUT_DIR, prev_model_name))
                
                self._saved_model_name = model_name
                self._best_AP = current_AP
            else: 
                if self._saved_model_name != prev_model_name:
                    # remove worse model
                    self._logger.info("AP did not increase, removing previous model: " + prev_model_name)
                    os.remove(os.path.join(self._cfg.OUTPUT_DIR, prev_model_name))

            if is_final:
                # correct the last_checkpoint file
                save_file = os.path.join(self._cfg.OUTPUT_DIR, "last_checkpoint")
                with open(save_file, 'w') as f:
                    f.write(self._saved_model_name)
