from roboflow import Roboflow
from ultralytics import YOLO
import ultralytics
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import io
import pandas as pd
import numpy as np

from typing import Optional


# Initializing model
model_obb = YOLO("./models/obb_model.pt")

def get_image_from_bytes(binary_image: bytes) -> Image:
    """Convert image from bytes to PIL RGB format
    
    Args:
        binary_image (bytes): The binary representation of the image
    
    Returns:
        PIL.Image: The image in PIL RGB format
    """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image


def get_bytes_from_image(image: Image) -> bytes:
    """
    Convert PIL image to Bytes
    
    Args:
    image (Image): A PIL image instance
    
    Returns:
    bytes : BytesIO object that contains the image in JPEG format with quality 85
    """
    return_image = io.BytesIO()
    image.save(return_image, format='JPEG', quality=85)  # save the image in JPEG format with quality 85
    return_image.seek(0)  # set the pointer to the beginning of the file
    return return_image

def get_json_prediction_result(input_image: Image):
    print(model_obb.model.names)
    # predict_result = model_obb.predict(
    #     imgsz=640,
    #     source=input_image,
    #     conf=0.25,
    #     save=True
    # )
    # # print(predict_result[0].to("cpu").obb)
    # print(predict_result[0])
    # print("==========================")
    # print(predict_result[0].obb)
    return {}