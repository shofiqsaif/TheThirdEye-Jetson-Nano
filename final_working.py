import threading
from threading import Thread
import time


import cv2
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


#Following is the pipeline for image capturing with cam
def gstreamer_pipeline(
    capture_width=224,
    capture_height=224,
    display_width=224,
    display_height=224,
    framerate=4,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


#The Following function keeps capturing photoes at 224*224*1
def captureFrames() :
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    it = 0
    f = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Display the resulting frame
        cv2.imshow('frame',gray)
        #it=it+1

        f= f+1    
        cv2.imwrite('/home/ghost/thesis/cam/pic/'+str(f)+'.jpg',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


#The following keeps printing work.
def detect():
    time.sleep(5)
    feature_seq = list()            #This will containg[1,20,7*7*1280]
    starting_frame = 1

    i = 1
    while True:
        seq = list() 
                                   #List of frame

        #Try to load current image
        try:
            img = image.load_img('pic/{}.jpg'.format(i), target_size=(224, 224))
        except:
            time.sleep(2)
            img = image.load_img('pic/{}.jpg'.format(i), target_size=(224, 224))

        i++         # Next image
        img = image.img_to_array(img,dtype=np.float32)
        seq.append(img)
        seq = np.array(seq,dtype=np.float32)

        interpreter1.set_tensor(input_details1[0]["index"], seq)   # seq is the input
        interpreter1.invoke()
        feature = interpreter1.get_tensor(output_details1[0]["index"])  


        print("feature {} extracted".format(i))
        feature = feature.reshape(-1)
        if len(feature_seq)==20:
            print("20 element in feature_seq")
            feature_seq.pop(0)
        feature_seq.append(feature)

        if i >=20 and i%4==0:  
            feature_seq_np = np.array(feature_seq,dtype=np.float32)
            feature_seq_np = feature_seq_np.reshape(-1,20,7*7*1280)
            print(feature_seq_np.shape)


            interpreter2.set_tensor(input_details2[0]["index"], feature_seq_np)   # seq is the input
            interpreter2.invoke()
            activity = interpreter2.get_tensor(output_details2[0]["index"]) 

            #print("Work at frame {} is ".format(i), [np.argmax(activity)])
            print("Work at frame {} is ".format(i), label[np.argmax(activity)])
            #print(activity)
            f = open("work.txt", "a")
            f.write("Work at frame {} is \n".format(i), label[np.argmax(activity)])
            f.close()
            
            time.sleep(1)



#Start of MAIN

try:
    cam = Thread(target=captureFrames)
    activity_recognizer = Thread(target = detect)


    cam.start()
    activity_recognizer.start()
except :
    print("An error has occured!")
