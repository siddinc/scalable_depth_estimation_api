import requests
import cv2
import json


# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"

# prepare headers for http request
content_type = "image/jpeg"
headers = {"content-type": content_type}

# define a video capture object
vid = cv2.VideoCapture(0)

ret, frame = vid.read()
# encode image as jpeg
_, img_encoded = cv2.imencode(".jpg", frame)

# send http request with image and receive response
response = requests.post(
    KERAS_REST_API_URL, data=img_encoded.tobytes(), headers=headers
)

# decode response
# print(json.load(response))
print(response.status_code)

# while True:
#     # Capture the video frame
#     # by frame
#     ret, frame = vid.read()
#     # encode image as jpeg
#     _, img_encoded = cv2.imencode(".jpg", frame)

#     # send http request with image and receive response
#     response = requests.post(KERAS_REST_API_URL, data=img_encoded.tostring(), headers=headers)

#     # the 'q' button is set as the
#     # quitting button you may use any
#     # desired button of your choice
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break


# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
