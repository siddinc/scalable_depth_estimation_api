import requests
import cv2
import json
from utils import base64_decode_image
import constants


# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://127.0.0.1:5000/predict"

# prepare headers for http request
# content_type = "image/jpg"
# headers = {"content-type": content_type}

# define a video capture object
vid = cv2.VideoCapture(0)

# ret, frame = vid.read()
# # encode image as jpeg
# _, img_encoded = cv2.imencode(".jpg", frame)


# payload = {"image": img_encoded}

# # send http request with image and receive response
# response = requests.post(KERAS_REST_API_URL, files=payload)
# print(response.json())
# decode response
# print(json.load(response))
# print(response.status_code)

while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    # encode image as jpeg
    _, img_encoded = cv2.imencode(".jpg", frame)
    payload = {"image": img_encoded}

    # send http request with image and receive response
    response = requests.post(KERAS_REST_API_URL, files=payload).json()
    if response["success"]:
        img = base64_decode_image(
            response["predictions"],
            constants.IMAGE_DTYPE,
            (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH),
        )
        # img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imshow("depth maps", img)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
