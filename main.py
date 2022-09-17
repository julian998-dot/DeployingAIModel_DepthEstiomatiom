import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import keras
import cv2
import pandas as pd
import os
import numpy as np
from ast import Break
from locale import normalize
import torch
import time
from scipy import zeros, signal, random


model_dir = os.path.join('model_tesis')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
model = tf.keras.models.load_model(model_dir)
model.summary()
class_names = ['epitrix', 'sano', 'tizon']


model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS",model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transforms
else:
    transform = midas_transforms.small_transform

cap = cv2.VideoCapture(0)
data = 0
b = signal.firwin(150, 0.004)
z = signal.lfilter_zi(b, 1) * data[0]
result = []
resultoNF = []
while cap.isOpened():
    success, img = cap.read()
    start = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ##########
    img2 = cv2.resize(img, (224, 224), interpolation = cv2.INTER_NEAREST)
    img2 = img2.reshape(224,224,3)
    img_array = tf.keras.utils.img_to_array(img2)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    #print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
    
    #############
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map =  prediction.cpu().numpy()
    #print(depth_map.mean())
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_64F)
    print(depth_map.mean())
    mean_value = round(depth_map.mean(),4)
    
    
    end = time.time()
    totalTime = end-start
    fps = 1 / totalTime
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
    
    cv2.putText(img, f"FPS: {int(fps)}",(20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.putText(img, f"{class_names[np.argmax(score)]}"+f": {round(100 * np.max(score),2)}",(20,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Image', img)
    cv2.imshow('Depth Map', depth_map)
    
    if cv2.waitKey(5) & 0XFF == ord('q'):
        break;
cap.release()
