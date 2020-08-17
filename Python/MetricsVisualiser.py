import matplotlib.pyplot as plt
import os
import json

class MetricsVisualiser:
    def visualiseMetrics(self, root_dir, network_type, type_of_annotation):
        json_pathname_extension = "-".join(str(elem) for elem in type_of_annotation)

        model = network_type + "-" + json_pathname_extension

        metrics = []
        with open(os.path.join(root_dir, "Models", model, "metrics.json"), "r") as f:
            for line in f:
                metrics.append(json.loads(line))

        scalars = []

        if len(type_of_annotation) > 1:
            for annotation in type_of_annotation:
                scalars.append("bbox/AP-" + annotation)

        scalars += ["iteration", "bbox/AP", "bbox/AP50", "bbox/AP75", "fast_rcnn/cls_accuracy", 
                    "fast_rcnn/false_negative", "loss_cls", "total_loss", "validation_loss", "lr"]

        for scalar in scalars:
            if scalar == "iteration":
                continue
            fig = plt.figure(figsize=(10,5))
            axes = fig.add_axes([.25,.25,.75,.75])

            plt.plot(
                [x["iteration"] for x in metrics if scalar in x],
                [x[scalar] for x in metrics if scalar in x]
            )
            plt.legend([scalar], loc="best")

            plt.show()

        fig = plt.figure(figsize=(10,5))
        axes = fig.add_axes([.25,.25,.75,.75])
        plt.plot(
            [x['iteration'] for x in metrics], 
            [x['total_loss'] for x in metrics])
        plt.plot(
            [x['iteration'] for x in metrics if 'validation_loss' in x], 
            [x['validation_loss'] for x in metrics if 'validation_loss' in x])
        plt.legend(['total_loss', 'validation_loss'], loc='upper right')
