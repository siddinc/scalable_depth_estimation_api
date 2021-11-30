from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json
import base64
import sys
import numpy as np


def base64_encode_image(a):
    return base64.b64encode(a).decode("utf-8")


def base64_decode_image(a, dtype, shape):
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)
    return a


def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype("float32")
    image = (image - image.min()) / (image.max() - image.min())
    return image


def load_model(model_pth: str, weights_path: str):
    with open(model_pth, "r") as f:
        m = f.read()
        model = model_from_json(m)
        model.load_weights(weights_path)
    return model
