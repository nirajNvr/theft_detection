from flask import Flask, render_template, request, jsonify, Response
import os
from webcam_app import process_webcam
import threading
import cv2
import numpy as np

app = Flask(__name__)

# Store device configurations
DEVICE_CONFIGS = {
    "device1": {
        "passcode": "1234",
        "ip": "http://192.168.0.61:8080/video"
    }
    # Add more devices as needed
}

# Global variable to store the current video stream
current_stream = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream_page():
    return render_template('stream.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global current_stream
        while True:
            if current_stream is not None:
                ret, frame = current_stream.read()
                if ret:
                    # Convert frame to JPEG
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                else:
                    break
            else:
                break

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/verify_device', methods=['POST'])
def verify_device():
    data = request.json
    device_id = data.get('device_id')
    passcode = data.get('passcode')
    
    if device_id in DEVICE_CONFIGS and DEVICE_CONFIGS[device_id]['passcode'] == passcode:
        return jsonify({
            'success': True,
            'ip': DEVICE_CONFIGS[device_id]['ip']
        })
    return jsonify({
        'success': False,
        'message': 'Invalid device ID or passcode'
    })

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global current_stream
    data = request.json
    ip_address = data.get('ip_address')
    
    # Initialize the video stream
    current_stream = cv2.VideoCapture(ip_address)
    if not current_stream.isOpened():
        return jsonify({
            'success': False,
            'message': 'Could not open video stream'
        })
    
    # Start webcam processing in a separate thread
    thread = threading.Thread(target=process_webcam, args=(ip_address,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 