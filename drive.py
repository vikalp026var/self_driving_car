import base64
import traceback
from io import BytesIO

import cv2
import eventlet
import numpy as np
import socketio
from flask import Flask
from PIL import Image
from tensorflow.keras.models import load_model

# Initialize Flask and SocketIO
sio = socketio.Server(logger=True, engineio_logger=True)
app = Flask(__name__)
model = load_model('self_car.h5', compile=False)
model.compile(optimizer='adam')

speed_limit = 10

# Image Preprocessing
def img_preprocessing(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200,66))
    img = img / 255
    return img

# Handle connection
@sio.on('connect')
def connect(sid, environ):
    # print(f'Connected with SID: {sid}')
    send_control(0, 0)

# Send control (steering and throttle)
def send_control(steering_angle, throttle):
    # print(f'Sending control - Steering Angle: {steering_angle}, Throttle: {throttle}')
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

# Handle telemetry
@sio.on('telemetry')
def telemetry(sid, data):
    try:
        if 'image' in data:
            speed = float(data['speed'])
            print(f'Received speed: {speed}')
            image = Image.open(BytesIO(base64.b64decode(data['image'])))
            image = np.asarray(image)
            image = img_preprocessing(image)
            image = np.array([image])
            steering_angle = float(model.predict(image)[0])
            throttle = 1.0 - speed / speed_limit
            print(f'SID: {sid}, Processed - Steering Angle: {steering_angle}, Throttle: {throttle}')
            send_control(steering_angle, throttle)
        else:
            print('Image data not received.')
    except Exception as e:
        print(f"Error processing telemetry data: {e}")
        traceback.print_exc()

# Wrap Flask application with SocketIO's middleware
app = socketio.Middleware(sio, app)

# Deploy as an eventlet WSGI server
if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
