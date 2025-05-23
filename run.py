import tensorflow
from tensorflow import keras
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagepath",type = str,required=True,help="provide the image path")

    args = parser.parse_args()

    imagepath = args.imagepath

    vgg16 = keras.applications.VGG16()

    image = keras.utils.load_img(imagepath,target_size = (224,224,3))
    input_arr = keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = vgg16.predict(input_arr)
    print(np.argmax(predictions))