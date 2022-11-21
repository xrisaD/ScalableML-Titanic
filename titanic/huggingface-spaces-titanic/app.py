import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def titanic(pclass, sex, age, fare):
    input_list = []
    input_list.append(pclass)
    input_list.append(sex)
    input_list.append(age)
    input_list.append(fare)
    # 'res' is a list of predictions returned as the label.
    print(np.asarray(input_list).reshape(1, -1))
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    image_url = "https://raw.githubusercontent.com/xrisaD/ScalableML-Titanic/main/images/" + str(res[0]) + ".jpg"
    img = Image.open(requests.get(image_url, stream=True).raw)
    return img #res[0]
        
demo = gr.Interface(
    fn=titanic,
    title="Titanic Survival Predictive Analytics",
    description="Experiment with titanic features to predict if passanger survived.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Dropdown(default=1, label="Passanger Class", choices = [1, 2, 3]),
        gr.inputs.Dropdown(default="Female", label="Sex", choices=["Female", "Male"], type="index"),
        gr.inputs.Dropdown(default="0-21", label="Age", choices=['0-21', '22-25', '26-40', '41-80', 'unknown'], type="index"),
        gr.inputs.Slider(minimum=0, maximum=600, default=50, label="Fare"),
        ],
    outputs=gr.Image(type="pil")) #gr.inputs.Number(label="Survived"))

demo.launch()

