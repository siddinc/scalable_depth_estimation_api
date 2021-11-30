# import the necessary packages
from utils import base64_decode_image, base64_encode_image, prepare_image
import requests
import sys
import cv2
import constants
import numpy as np
from PIL import Image

# from VideoCapture import VideoStream
# import time


KERAS_REST_API_URL = "http://127.0.0.1:5000/predict"

vid = cv2.VideoCapture(0)

# vid = VideoStream(0, 120).start()
# time.sleep(1)

# while vid.more():
while True:
    ret, frame = vid.read()
    # frame = vid.read()
    # _, img_encoded = cv2.imencode(".jpg", frame)
    frame = prepare_image(
        Image.fromarray(frame), (constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT)
    )
    frame = frame.copy(order="C")
    img_encoded = base64_encode_image(frame)
    payload = {"image": img_encoded}

    response = requests.post(KERAS_REST_API_URL, json=payload).json()
    if response["success"]:
        img = base64_decode_image(
            response["predictions"],
            constants.IMAGE_DTYPE,
            (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH),
        )
        # img = (cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC) * 255).astype(
        #     np.uint8
        # )
        img = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_HSV)
        cv2.imshow("depth maps", img)
    # cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# vid.stop()
vid.release()
cv2.destroyAllWindows()
