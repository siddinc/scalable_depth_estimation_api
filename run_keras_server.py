from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions
from threading import Thread
from PIL import Image
from constants import (
    IMAGE_QUEUE, BATCH_SIZE,
    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANS, IMAGE_DTYPE,
    SERVER_SLEEP, CLIENT_SLEEP,
)
from utils import (
    base64_decode_image,
    base64_encode_image,
    prepare_image
)
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
            image = base64_decode_image(q["image"], IMAGE_DTYPE,
                                        (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANS))
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
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image = image.copy(order="C")
            k = str(uuid.uuid4())
            d = {"id": k, "image": base64_encode_image(image)}
            db.rpush(IMAGE_QUEUE, json.dumps(d))

            while True:
                output = db.get(k)

                if output is not None:
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)
                    db.delete(k)
                    break
                time.sleep(CLIENT_SLEEP)
            data["success"] = True
    return flask.jsonify(data)


if __name__ == "__main__":
    print("* Starting model service...")
    t = Thread(target=classify_process, args=())
    t.daemon = True
    t.start()
    print("* Starting web service...")
    app.run()
