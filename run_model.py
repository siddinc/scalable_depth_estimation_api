from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions
import numpy as np
import constants
import utils
import redis
import time
import json


db = redis.StrictRedis(host=constants.REDIS_HOST,
                       port=constants.REDIS_PORT, db=constants.REDIS_DB)


def classify_process():
    print("* Loading model...")
    model = ResNet50(weights="imagenet")
    print("* Model loaded")

    while True:
        queue = db.lrange(constants.IMAGE_QUEUE, 0,
                          constants.BATCH_SIZE - 1)
        imageIDs = []
        batch = None

        for q in queue:
            q = json.loads(q.decode("utf-8"))
            image = utils.base64_decode_image(q["image"],
                                              constants.IMAGE_DTYPE,
                                              (1, constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH,
                                               constants.IMAGE_CHANS))
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
            db.ltrim(constants.IMAGE_QUEUE, len(imageIDs), -1)
        time.sleep(constants.SERVER_SLEEP)


if __name__ == "__main__":
    classify_process()
