import torch
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import BoxMode
from detectron2.structures.boxes import Boxes

class CustomVisualizer(Visualizer):
    def _create_text_labels(self, classes, scores, class_names):
        """
        Args:
            classes (list[int] or None):
            scores (list[float] or None):
            class_names (list[str] or None):
        Returns:
            list[str] or None
        """
        labels = None
        if classes is not None and class_names is not None and len(class_names) > 1:
            labels = [class_names[i] for i in classes]
        if scores is not None:
            if labels is None:
                labels = ["{:.0f}%".format(s * 100) for s in scores]
            else:
                labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
        return labels

    def draw_dataset_dict(self, dic, category=None):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.

        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.
            category: the integer category for the desired annotation to display as a list or None if all of them

        Returns:
            output (VisImage): image object with visualizations.
        """
        # start additional code
        unfiltered_annos = dic.get("annotations", None)
        if category == None:
            annos = unfiltered_annos
        else:
            annos = [] 
            for annotations in unfiltered_annos:
                if annotations["category_id"] in category:
                    annos.append(annotations)
        # end additional code

        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS) for x in annos]

            labels = [x["category_id"] for x in annos]
            colors = None
            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                colors = [
                    self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in labels
                ]
            names = self.metadata.get("thing_classes", None)
            if names:
                labels = [names[i] for i in labels]
            labels = [
                "{}".format(i) + ("|crowd" if a.get("iscrowd", 0) else "")
                for i, a in zip(labels, annos)
            ]
            self.overlay_instances(
                labels=labels, 
                boxes=boxes, 
                masks=masks, 
                keypoints=keypts, 
                assigned_colors=colors,
                alpha=1000.0 # added alpha to be 1000.0
            )

        sem_seg = dic.get("sem_seg", None)
        if sem_seg is None and "sem_seg_file_name" in dic:
            with PathManager.open(dic["sem_seg_file_name"], "rb") as f:
                sem_seg = Image.open(f)
                sem_seg = np.asarray(sem_seg, dtype="uint8")
        if sem_seg is not None:
            self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)
        return self.output

    # might not work for every useage - but works for me
    # have not tested with keypoints and masks since my model does not have these
    def draw_instance_predictions(self, predictions, category=None):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
            category: the integer category for the desired annotation to display as a list or None if all of them

        Returns:
            output (VisImage): image object with visualizations.
        """

        # start additional code
        if category == None:
            boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
            scores = predictions.scores if predictions.has("scores") else None
            classes = predictions.pred_classes if predictions.has("pred_classes") else None
            labels = self._create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
            keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        else:
            all_boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
            all_scores = predictions.scores if predictions.has("scores") else None
            all_classes = predictions.pred_classes if predictions.has("pred_classes") else None
            all_labels = self._create_text_labels(all_classes, all_scores, self.metadata.get("thing_classes", None))
            all_keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

            boxes = [] if all_boxes != None else None
            scores = [] if all_scores != None else None
            classes = [] if all_classes != None else None
            labels = [] if all_labels != None else None
            keypoints = [] if all_keypoints != None else None

            for c in category:
                for i in range(0, len(all_classes)):
                    if all_classes[i] == c:
                        classes.append(all_classes[i])

                        if all_boxes != None:
                            boxes.append(all_boxes[i])
                        if all_scores != None:
                            scores.append(all_scores[i])
                        if all_labels != None:
                            labels.append(all_labels[i])
                        if all_keypoints != None:
                            keypoints.append(all_keypoints[i])

            if boxes != None and len(boxes) > 0:
                boxes = Boxes(torch.cat([b.tensor for b in boxes], dim=0))
            if scores != None and len(scores) > 0:
                scores = torch.stack(scores)
            if classes != None and len(classes) > 0:
                classes = torch.stack(classes)
        # end additional code

        # removed alpha from here and put it as fixed value
        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None
        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
        else:
            colors = None
        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
            )

        self.overlay_instances(
            labels=labels,
            boxes=boxes,
            masks=masks,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=1000.0, # changed alpha to be 1000.0
        )
        return self.output