from utils import base64_decode_image
import requests
import cv2
import constants
import numpy as np


KERAS_REST_API_URL = "http://127.0.0.1:5000/predict"

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    _, img_encoded = cv2.imencode(".jpg", frame)
    payload = {"image": img_encoded}

    response = requests.post(KERAS_REST_API_URL, files=payload).json()
    if response["success"]:
        img = base64_decode_image(
            response["predictions"],
            constants.IMAGE_DTYPE,
            (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH),
        )
        img = (cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
               * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
        cv2.imshow("depth maps", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
