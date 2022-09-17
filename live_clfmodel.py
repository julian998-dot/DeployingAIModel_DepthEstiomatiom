import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import keras
import cv2
import pandas as pd
import os
import numpy as np

def cnn_model():
    model_dir = os.path.join('model_tesis')
    print(model_dir)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    model = tf.keras.models.load_model(model_dir)
    model.summary()
    vid = cv2.VideoCapture(0)
    class_names = ['epitrix', 'sano', 'tizon']

    while(True):
        ret, frame = vid.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_NEAREST)
        img = img.reshape(224,224,3)
        cv2.imshow('frame', frame)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
        #predict = np.argmax(predictions[0])
        #print(predict)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
def main():
    cnn_model()
    
if __name__ == "__main__":
    main()