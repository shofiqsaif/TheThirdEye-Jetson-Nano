import threading
from threading import Thread
import time
import timehandler


import cv2
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


#Loading the labels
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

#This is the time handler we will use
th = timehandler.DateTimeHandler()

#Just print the times for testing or log
th.printDateTimeBeautifully(th.current_time)
th.printDateTimeBeautifully(th.capture_time)
th.printDateTimeBeautifully(th.feature_time)



#Following is the pipeline for image capturing with cam with defined {capture_width,capture_hight} and defined framerate.
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
        #Write the frame in year_month_day_hour_munite_second_frameNumber format.   
        cv2.imwrite('/home/ghost/thesis/cam/pic/'+th.returnDateTimeString(th.capture_time)+'_'+str(f)+'.jpg',gray)
        if f == 4:
            f = 0
            th.getNextSecond(th.capture_time)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def frameToFeature(img):
    """ This function takes a image (224*224*3) as input and extract the feature of the image by passing is through the mobilenetV2. Later it flatten the features and return that.  """
    seq = list()
    img = image.img_to_array(img,dtype=np.float32)
    seq.append(img)
    seq = np.array(seq,dtype=np.float32)

    interpreter1.set_tensor(input_details1[0]["index"], seq)   # seq is the input
    interpreter1.invoke()
    feature = interpreter1.get_tensor(output_details1[0]["index"])
    #print("feature {} extracted".format(i))
    feature = feature.reshape(-1)
    return feature


#The following keeps printing work.
def detect():
    """ Upon calling this function. It keeps detecting work done. """

    #By sleeping at starts, it gives some time to the capture function to capture frames even before calling the detect function.
    time.sleep(10)
    feature_seq = list()            #This will containg[1,20,7*7*1280] before bassing to the LSTM based model.
    

    while True:

        # get the exact moment/second we are going to detect the activities.
        current_detection_time = th.returnDateTimeString(th.feature_time)

        #Try to load all the 4 frames of current second/moment of detection_time.
        try:
            frame1 = image.load_img('pic/{}_{}.jpg'.format(current_detection_time,1), target_size=(224, 224))
            frame2 = image.load_img('pic/{}_{}.jpg'.format(current_detection_time,2), target_size=(224, 224))
            frame3 = image.load_img('pic/{}_{}.jpg'.format(current_detection_time,3), target_size=(224, 224))
            frame4 = image.load_img('pic/{}_{}.jpg'.format(current_detection_time,4), target_size=(224, 224))
        except:
            time.sleep(2)
            continue


        # Extracts features from currently loaded frames.
        feature_of_frame1 = frameToFeature(frame1)
        feature_of_frame2 = frameToFeature(frame2)
        feature_of_frame3 = frameToFeature(frame3)
        feature_of_frame4 = frameToFeature(frame4)
        
        
        # If feature_seq is 20 that means we have feature_frames from past 5 second and ready to detect some activity from it .
        if len(feature_seq)==20:
            #print("20 element in feature_seq")

            #As we have to pass the feature_sequence as numpy array to the LSTM model, we are converting the list to numpy array
            feature_seq_np = np.array(feature_seq,dtype=np.float32)
            feature_seq_np = feature_seq_np.reshape(-1,20,7*7*1280) # Reshaping to add a extra dimention/batch_dimention to the numpy array
            print(feature_seq_np.shape)


            # Detecting Activity
            interpreter2.set_tensor(input_details2[0]["index"], feature_seq_np)   # seq is the input
            interpreter2.invoke()
            activity = interpreter2.get_tensor(output_details2[0]["index"]) 

            

            print(np.argmax(activity))
            print("Work at time {} is {}".format(current_detection_time, label[np.argmax(activity)]))

            #Writing the detected activity to a file
            f = open("work.txt", "a")
            f.write("Work at time {} is {}\n".format(current_detection_time, label[np.argmax(activity)]))
            f.close()
            
            time.sleep(1)
            #Removing first 4 frames as they are done working on.
            feature_seq = feature_seq[4:]

        feature_seq.append(feature_of_frame1)
        feature_seq.append(feature_of_frame2)
        feature_seq.append(feature_of_frame3)
        feature_seq.append(feature_of_frame4)

        # Getting the next feature_time to work on.
        th.getNextSecond(th.feature_time)
        



#Start of MAIN

try:
    cam = Thread(target=captureFrames)
    activity_recognizer = Thread(target = detect)


    cam.start()
    activity_recognizer.start()
except :
    print("An error has occured!")
