import requests
from utils import base64_decode_image, prepare_image, base64_encode_image
from flask_cors import CORS
from PIL import Image
import constants
import numpy as np
import flask
import redis
import uuid
import time
import json
import io
import cv2


app = flask.Flask(__name__)
CORS(app)
db0 = redis.StrictRedis(
    host=constants.REDIS_HOST, port=constants.REDIS_PORT, db=constants.REDIS_DB0
)
db1 = redis.StrictRedis(
    host=constants.REDIS_HOST, port=constants.REDIS_PORT, db=constants.REDIS_DB1
)

# vid = cv2.VideoCapture(0)


@app.route("/")
def homepage():
    return flask.render_template("index.html")


# def gen_frames():
#     while True:
#         ret, frame = vid.read()
#         frame = prepare_image(
#             Image.fromarray(frame), (constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT)
#         )
#         frame = frame.copy(order="C")
#         img_encoded = base64_encode_image(frame)
#         k = str(uuid.uuid4())
#         d = {"id": k, "image": img_encoded}
#         db0.rpush(constants.IMAGE_QUEUE, json.dumps(d))
#         while True:
#             output = db1.get(k)
#             if output is not None:
#                 output = output.decode("utf-8")
#                 img = base64_decode_image(
#                     json.loads(output),
#                     constants.IMAGE_DTYPE,
#                     (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH),
#                 )
#                 db1.delete(k)
#                 break
#         ret, buffer = cv2.imencode(".jpg", img)
#         frame = buffer.tobytes()
#         yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


# @app.route("/video_feed")
# def video_feed():
#     return flask.Response(
#         gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
#     )


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        # print(flask.request.data)
        if flask.request.json["image"]:
            # image = flask.request.files["image"].read()
            # image = Image.open(io.BytesIO(image))
            # image = base64_decode_image(
            #     flask.request.files["image"],
            #     constants.IMAGE_DTYPE,
            #     (constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT, constants.IMAGE_CHANS),
            # )
            # image = prepare_image(
            #     image, (constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT)
            # )
            # image = image.copy(order="C")
            # image = flask.request.files["image"].read()
            # image = io.BytesIO(image)
            # print(image)
            # print(type(image))
            image = flask.request.json["image"]
            k = str(uuid.uuid4())
            # image = base64_encode_image(image)
            d = {"id": k, "image": image}
            db0.rpush(constants.IMAGE_QUEUE, json.dumps(d))

            while True:
                output = db1.get(k)

                if output is not None:
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)
                    db1.delete(k)
                    break

                time.sleep(constants.CLIENT_SLEEP)
            data["success"] = True
    return flask.jsonify(data)


if __name__ == "__main__":
    print("* Starting web service...")
    app.run(debug=True)
