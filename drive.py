#parsing command line arguments
import argparse
#decoding camera images
import base64
#for frametimestamp saving
from datetime import datetime
#reading and writing files
import os
#high level file operations
import shutil
#matrix math
import numpy as np
#real-time server
import socketio
#concurrent networking 
import eventlet
#web server gateway interface
import eventlet.wsgi
#image manipulation
from PIL import Image
#web framework
from flask import Flask
#input output
from io import BytesIO
import cv2
#load our saved model
from keras.models import load_model



#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__) #'__main__'
#init our model and image array as empty


#set min/max speed for our autonomous car
MAX_SPEED = 10

def preprocess(img):
    img=img[60:135,:,:]
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV) # Lnaes become more visible
    img-cv2.GaussianBlur(img,(3,3),0)
    img=cv2.resize(img,(200,66))
    img=img/255 # normalization 0 to 1
    return img

#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
   
    # The current speed of the car
    speed = float(data["speed"])
    # The current image from the center camera of the car
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
        
    image = np.asarray(image)       # from PIL image to numpy array
    image = preprocess(image) # apply the preprocessing
    image = np.array([image])       # the model expects 4D array

    # predict the steering angle for the image
    steering= float(model.predict(image))
    # lower the throttle as the speed increases
    # if the speed is above the current speed limit, we are on a downhill.
    # make sure we slow down first and then go back to the original max speed.

    throttle = 1.0 - speed/MAX_SPEED
    if throttle < 0:
        throttle=0

    print('{} {} {}'.format(steering, throttle, speed))
    send_control(steering, throttle)
      

    



@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering.__str__(),
            'throttle': throttle.__str__()
        },
        )


if __name__ == '__main__':
  

    #load model
    model = load_model('model.h5')

     
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)