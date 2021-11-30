from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions
from threading import Thread
from PIL import Image
from constants import (
    IMAGE_QUEUE,
    BATCH_SIZE,
    IMAGE_DTYPE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANS,
    SERVER_SLEEP,
    CLIENT_SLEEP,
    MODEL_PTH,
    WEIGHTS_PTH,
)
from utils import base64_decode_image, base64_encode_image, prepare_image, load_model

import numpy as np
import flask
import redis
import uuid
import time
import json
import io


app = flask.Flask(__name__)
db = redis.StrictRedis(host="localhost", port=6379, db=0)
model = None


def classify_process():
    print("* Loading model...")
    model = ResNet50(weights="imagenet")
    print("* Model loaded")

    while True:
        queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
        imageIDs = []
        batch = None

        for q in queue:
            q = json.loads(q.decode("utf-8"))
            image = base64_decode_image(
                q["image"], IMAGE_DTYPE, (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANS)
            )
            if batch is None:
                batch = image
            else:
                batch = np.vstack([batch, image])
            imageIDs.append(q["id"])

        if len(imageIDs) > 0:
            print("* Batch size: {}".format(batch.shape))
            preds = model.predict(batch)
            results = decode_predictions(preds)

            for (imageID, resultSet) in zip(imageIDs, results):
                output = []
                for (imagenetID, label, prob) in resultSet:
                    r = {"label": label, "probability": float(prob)}
                    output.append(r)
                db.set(imageID, json.dumps(output))
            db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)
        time.sleep(SERVER_SLEEP)


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format and prepare it for
            # classification
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            # ensure our NumPy array is C-contiguous as well,
            # otherwise we won't be able to serialize it
            image = image.copy(order="C")
            # generate an ID for the classification then add the
            # classification ID + image to the queue
            k = str(uuid.uuid4())
            d = {"id": k, "image": base64_encode_image(image)}
            db.rpush(IMAGE_QUEUE, json.dumps(d))

            while True:
                # attempt to grab the output predictions
                output = db.get(k)
                # check to see if our model has classified the input
                # image
                if output is not None:
                    # add the output predictions to our data
                    # dictionary so we can return it to the client
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)
                    # delete the result from the database and break
                    # from the polling loop
                    db.delete(k)
                    break
                # sleep for a small amount to give the model a chance
                # to classify the input image
                time.sleep(CLIENT_SLEEP)
            # indicate that the request was a success
            data["success"] = True
        # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    print("* Starting model service...")
    t = Thread(target=classify_process, args=())
    t.daemon = True
    t.start()
    print("* Starting web service...")
    app.run()
