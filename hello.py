import constants
import matplotlib.pyplot as plt
import numpy as np
import utils
import json


with open("./hello.json", "r") as f:
    data = json.load(f)


if __name__ == "__main__":
    s = data["predictions"]
    x = utils.base64_decode_image(s, constants.IMAGE_DTYPE,
                                  (128, 128))

    plt.imshow(x, cmap="gray")
    plt.show()
