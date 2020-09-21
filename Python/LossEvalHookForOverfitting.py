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

# several tests have shown that this hook does not produce better results
# which is why the simpler hook, will be used in the trainer

# this Hook will check for the best loss instead of best accuracy
# after 5x val_period when mean_loss was < self._best_loss - self._threshold every time
# the algorithm is going to lower the learning rate by cfg.SOLVER.GAMMA
# if the next 5x val_periods will also not provide a better loss, the training will stop
# else after_reduce_lr_failed_to_get_better_count will become 0 again and the learning rate will again
# be reduced by cfg.SOLVER.GAMMA when the loss wont lower sigificantly for 5x val_periods
class LossEvalHookForOverfitting(HookBase):
    def __init__(self, cfg, val_period, model, scheduler, data_loader):
        self._threshold = 0.005 # this threshold has to be surpassed for the mean_loss
        self._best_loss = math.inf
        self._best_AP = 0
        self._loss_failed_to_get_better_count = 0
        self._after_reduce_lr_failed_to_get_better_count = 0
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
        
    def _initialize(self):
        for files in os.listdir(self._cfg.OUTPUT_DIR):
            # if there are earlier models - get logged variables from metrics
            if "model" in files:
                self._saved_model_name = files

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

                prev_losses = []
                for x in lines:
                    if "validation_loss" in x:
                        prev_losses.append(x['validation_loss'])
                    if "loss_failed_to_get_better_count" in x:
                        self._loss_failed_to_get_better_count = x["loss_failed_to_get_better_count"]
                    if "after_reduce_lr_failed_to_get_better_count" in x:
                        self._after_reduce_lr_failed_to_get_better_count = x["after_reduce_lr_failed_to_get_better_count"]
                    if "lr" in x:
                        self._cfg.SOLVER.BASE_LR = x["lr"]
                    if "bbox/AP" in x:
                        self._best_AP = x["bbox/AP"]

                as_strings = ["%.5f" % e for e in prev_losses]
                # base _best_loss acording to stored failed_counter because prev_losses[-1] does not actually have to be the best loss
                self._best_loss = prev_losses[-1 - int(self._loss_failed_to_get_better_count)]

                self._logger.info("found existing model: " + self._saved_model_name)
                self._logger.info("previous losses: " + ', '.join(as_strings))
                self._logger.info("best_loss: " + str(self._best_loss))
                self._logger.info("loss_failed_to_get_better_count: " + str(self._loss_failed_to_get_better_count))
                self._logger.info("after_reduce_lr_failed_to_get_better_count: " + str(self._after_reduce_lr_failed_to_get_better_count))

    def after_step(self):
        if not self._is_initialized:
            self._initialize()
            self._is_initialized = True
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
                    save_file = os.path.join(self._cfg.OUTPUT_DIR, "last_checkpoint")
                    with open(save_file, 'w') as f:
                        f.write(self._saved_model_name)

                    if self._after_reduce_lr_failed_to_get_better_count >= 2:
                        raise Exception("Could not compute a better loss - training stopped as a result")
                    else:
                        get_event_storage().put_scalar("after_reduce_lr_failed_to_get_better_count", int(self._after_reduce_lr_failed_to_get_better_count), smoothing_hint=False)

                        self._logger.info("Could not compute a better loss for the last 5 iterations - lowering learning rate in the next iteration")
                        
                        # reset the loss failed counter
                        self._loss_failed_to_get_better_count = 0
                        # ajust the learning rate
                        self._scheduler.base_lrs = (self._scheduler.base_lrs[-1] * self._cfg.SOLVER.GAMMA, ) # seems like a nasty hack to me, but works like a charm!

            get_event_storage().put_scalar("loss_failed_to_get_better_count", int(self._loss_failed_to_get_better_count), smoothing_hint=False)
            model_name = "model_" + ("%07.0f" % self.trainer.iter) + ".pth"

            current_AP = self.trainer.storage._latest_scalars["bbox/AP"][0]
            self._logger.info("best AP: " + str(self._best_AP) + ", current AP: " + str(current_AP))
            if self._saved_model_name is None:
                self._saved_model_name = model_name
                self._best_AP = current_AP
            elif current_AP > self._best_AP:
                # remove old model and save new model
                self._logger.info("found model with better AP, saving model : " + model_name)
                self._logger.info("removing previous model: " + self._saved_model_name)
                os.remove(os.path.join(self._cfg.OUTPUT_DIR, self._saved_model_name))
                self._saved_model_name = model_name
                self._best_AP = current_AP
            else: 
                # remove worse model
                self._logger.info("AP did not increase, removing current model: " + model_name)
                os.remove(os.path.join(self._cfg.OUTPUT_DIR, model_name))

            # correct the last_checkpoint file
            save_file = os.path.join(self._cfg.OUTPUT_DIR, "last_checkpoint")
            with open(save_file, 'w') as f:
                f.write(self._saved_model_name)
