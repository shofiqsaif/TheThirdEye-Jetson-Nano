import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

label = np.load('label.npy')
label = np.unique(label)


#This is interpreter for MobileNetV2_with_Preprocessing
interpreter1 = tf.lite.Interpreter(model_path='mobilenetv2withpreprocessing.tflite')
interpreter1.allocate_tensors()
input_details1 = interpreter1.get_input_details()
output_details1 = interpreter1.get_output_details()

#This is interpreter for GRU_8_8
interpreter2 = tf.lite.Interpreter(model_path='gru.tflite')
interpreter2.allocate_tensors()
input_details2 = interpreter2.get_input_details()
output_details2 = interpreter2.get_output_details()


feature_seq = list()			#This will containg[1,20,7*7*1280]

for i in range(1,21):
	seq = list()							#List of frame
	img = image.load_img('pic/{}.jpg'.format(i), target_size=(224, 224))
	img = image.img_to_array(img,dtype=np.float32)
	seq.append(img)
	seq = np.array(seq,dtype=np.float32)

	interpreter1.set_tensor(input_details1[0]["index"], seq)   # seq is the input
	interpreter1.invoke()
	feature = interpreter1.get_tensor(output_details1[0]["index"])	


	print("feature {} extracted".format(i))
	feature = feature.reshape(-1)
	feature_seq.append(feature)

feature_seq = np.array(feature_seq,dtype=np.float32)
feature_seq = feature_seq.reshape(-1,20,7*7*1280)
print(feature_seq.shape)

interpreter2.set_tensor(input_details2[0]["index"], feature_seq)   # seq is the input
interpreter2.invoke()
activity = interpreter2.get_tensor(output_details2[0]["index"])	

print(label[np.argmax(activity)])

	






