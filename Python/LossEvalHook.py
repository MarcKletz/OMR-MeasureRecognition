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

# utils.logger does not work on windows - log_first_n or log_every_n both not working
# using logger.info instead
class LossEvalHook(HookBase):
    def __init__(self, cfg, val_period, model, scheduler, data_loader):
        self._threshold = 0.005 # this threshold has to be surpassed for the mean_loss
        self._best_loss = math.inf
        self._loss_failed_to_get_better_count = 0
        self._after_reduce_lr_failed_to_get_better_count = 0
        self._saved_model_names = []
        self._scheduler = scheduler

        self._cfg = cfg
        self._period = val_period
        self._checkpointer_dir = cfg.OUTPUT_DIR
        self._model = model
        self._data_loader = data_loader
        self._logger = logging.getLogger("detectron2")

        # in case there are earlier models in the checkpoint dir - load them
        for files in os.listdir(cfg.OUTPUT_DIR):
            if "model_" in files:
                self._saved_model_names.append(files)

        self._saved_model_names = sorted(self._saved_model_names)

        # if there are earlier models - get logged variables from metrics
        if len(self._saved_model_names) > 0:
            lines = []
            with open(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), 'r') as f:
                for line in f:
                    lines.append(json.loads(line))

            prev_losses = []
            for x in lines:
                if "validation_loss" in x:
                    prev_losses.append(x['validation_loss'])
                if "loss_failed_to_get_better_count" in x:
                    self._loss_failed_to_get_better_count = x["loss_failed_to_get_better_count"]
                if "after_reduce_lr_failed_to_get_better_count" in x:
                    self._after_reduce_lr_failed_to_get_better_count = x["after_reduce_lr_failed_to_get_better_count"]
                if "reduced_lr" in x:
                    self._cfg.SOLVER.BASE_LR = x["reduced_lr"]

            as_strings = ["%.5f" % e for e in prev_losses]
            # base _best_loss acording to stored failed_counter because prev_losses[-1] does not actually have to be the best loss
            self._best_loss = prev_losses[-1 - int(self._loss_failed_to_get_better_count)]

            self._logger.info("found existing models")
            self._logger.info("previous losses: " + ', '.join(as_strings))
            self._logger.info("best_loss=" + str(self._best_loss))
            self._logger.info("loss_failed_to_get_better_count=" + str(self._loss_failed_to_get_better_count))
            self._logger.info("saved_model_names=" + ', '.join(self._saved_model_names))
            self._logger.info("after_reduce_lr_failed_to_get_better_count=" + str(self._after_reduce_lr_failed_to_get_better_count))

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
        self._logger.info("mean_loss=" + str(mean_loss) + ", best_loss=" + str(self._best_loss))
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
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            losses, mean_loss = self._do_loss_eval()
            if mean_loss < self._best_loss - self._threshold:
                self._best_loss = mean_loss
                self._loss_failed_to_get_better_count = 0
                self._after_reduce_lr_failed_to_get_better_count = 0
                get_event_storage().put_scalar("after_reduce_lr_failed_to_get_better_count", int(self._after_reduce_lr_failed_to_get_better_count), smoothing_hint=False)
            else:
                self._loss_failed_to_get_better_count += 1
                self._logger.info("could not find a better loss with mean_loss=" + str(mean_loss) + " best_loss=" + str(self._best_loss) + " and failed count=" + str(self._loss_failed_to_get_better_count))
                if self._loss_failed_to_get_better_count >= 5: # loss didnt get better the last 5 iterations                
                    self._after_reduce_lr_failed_to_get_better_count += 1

                    # correct the last_checkpoint file
                    save_file = os.path.join(self._checkpointer_dir, "last_checkpoint")
                    with open(save_file, 'w') as f:
                        f.write(self._saved_model_names[0])

                    if self._after_reduce_lr_failed_to_get_better_count >= 2:
                        raise Exception("Could not compute a better loss - training stopped as a result")
                    else:
                        get_event_storage().put_scalar("after_reduce_lr_failed_to_get_better_count", int(self._after_reduce_lr_failed_to_get_better_count), smoothing_hint=False)

                        self._logger.info("Could not compute a better loss for the last 5 iterations - lowering learning rate in the next iteration")
                        self._logger.info("removing the best model from _saved_model_names so that it wont get deleted: " + self._saved_model_names[0])
                        
                        # TODO: save and load the best model into a dict
                        # self._best_model_dict.update({str(self._saved_model_names[0]) : self._best_loss})
                        # self._logger.info("best models so far: " + str(self._best_model_dict))
                        
                        # keep the model where the loss was best and dont delete it
                        self._saved_model_names.pop(0)
                        # reset the loss failed counter
                        self._loss_failed_to_get_better_count = 0
                        # ajust the learning rate
                        self._scheduler.base_lrs = (self._scheduler.base_lrs[-1] * self._cfg.SOLVER.GAMMA, ) # seems like a nasty hack to me, but works like a charm!
                        get_event_storage().put_scalar("reduced_lr", self._scheduler.base_lrs[-1], smoothing_hint=False)

            get_event_storage().put_scalar("loss_failed_to_get_better_count", int(self._loss_failed_to_get_better_count), smoothing_hint=False)
            model_name = "model_" + ("%07.0f" % self.trainer.iter) + ".pth"
            self._saved_model_names.append(model_name)

            self._logger.info("saving model to: " + model_name + ", saved_model_names=" + ', '.join(self._saved_model_names))
            # remove old models that are no longer needed
            while len(self._saved_model_names) >= 6:
                self._logger.info("exeeded model save threshold - removing " + self._saved_model_names[0])
                os.remove(os.path.join(self._checkpointer_dir, self._saved_model_names[0]))
                self._saved_model_names.pop(0)
                self._logger.info("saved_model_names=" + ", ".join(self._saved_model_names))