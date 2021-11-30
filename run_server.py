from PIL import Image
import constants
import utils
import numpy as np
import flask
import redis
import uuid
import time
import json
import io


app = flask.Flask(__name__)
db = redis.StrictRedis(
    host=constants.REDIS_HOST, port=constants.REDIS_PORT, db=constants.REDIS_DB
)


@app.route("/")
def homepage():
    return "<H1>Welcome to the PyImageSearch Keras REST API!</H1>"


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    print(data)
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = utils.prepare_image(
                image, (constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT)
            )
            image = image.copy(order="C")
            k = str(uuid.uuid4())
            image = utils.base64_encode_image(image)
            d = {"id": k, "image": image}
            db.rpush(constants.IMAGE_QUEUE, json.dumps(d))

            while True:
                output = db.get(k)
                if output is not None:
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)
                    db.delete(k)
                    break
                time.sleep(constants.CLIENT_SLEEP)
            data["success"] = True
    return flask.jsonify(data)


if __name__ == "__main__":
    print("* Starting web service...")
    app.run()
