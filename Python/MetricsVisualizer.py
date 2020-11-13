import matplotlib.pyplot as plt
import streamlit as st
import os
import json

class MetricsVisualizer:
    def visualizeMetrics(self, root_dir, network_type, type_of_annotation, start_plot_iter=0):
        json_pathname_extension = "-".join(str(elem) for elem in type_of_annotation)

        model = network_type + "-" + json_pathname_extension

        metrics = []
        with open(os.path.join(root_dir, "Models", model, "metrics.json"), "r") as f:
            for line in f:
                l = json.loads(line)
                if "iteration" in l and l["iteration"] > start_plot_iter:
                    metrics.append(l)

        scalars = []

        if len(type_of_annotation) > 1:
            for annotation in type_of_annotation:
                scalars.append("bbox/AP-" + annotation)

        scalars += ["bbox/AP", "bbox/AP50", "bbox/AP75", "fast_rcnn/cls_accuracy", 
                    "fast_rcnn/false_negative", "loss_cls", "total_loss", "validation_loss", "lr"]

        for scalar in scalars:
            if st._is_running_with_streamlit:
                fig, axes = plt.subplots()
            fig = plt.figure(figsize=(10,5))
            axes = fig.add_axes([.25,.25,.75,.75])

            plt.plot(
                [x["iteration"] for x in metrics if scalar in x],
                [x[scalar] for x in metrics if scalar in x]
            )
            plt.legend([scalar], loc="best")
            if st._is_running_with_streamlit:
                st.pyplot(fig)
            else:
                plt.show()



        for x in metrics:
            if "total_loss" not in x:
                print("no total loss:", x)



        if st._is_running_with_streamlit:
            fig, axes = plt.subplots()
        fig = plt.figure(figsize=(10,5))
        axes = fig.add_axes([.25,.25,.75,.75])
        plt.plot(
            [x['iteration'] for x in metrics if 'total_loss' in x], 
            [x['total_loss'] for x in metrics if 'total_loss' in x])
        plt.plot(
            [x['iteration'] for x in metrics if 'validation_loss' in x], 
            [x['validation_loss'] for x in metrics if 'validation_loss' in x])
        plt.legend(['total_loss', 'validation_loss'], loc='upper right')
        if st._is_running_with_streamlit:
            st.pyplot(fig)
        else:
            plt.show()