from flask import Flask, render_template, Response, redirect, jsonify
from camera import VideoCamera

app = Flask(__name__)
camera = None
current_emotion = "Neutral"  # Global variable to hold current emotion

@app.route('/')
def index():
    global camera
    if not camera:
        camera = VideoCamera()
    return render_template('index.html', camera_on=True)


@app.route('/video_feed')
def video_feed():
    global camera
    def gen(camera):
        while True:
            frame = camera.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_feed')
def stop_feed():
    global camera
    if camera:
        del camera
        camera = None
    return redirect('/')


@app.route('/get_emotion')
def get_emotion():
    global camera
    if camera:
        return jsonify({'emotion': camera.emotion})
    return jsonify({'emotion': "Neutral"})

if __name__ == '__main__':
    app.run(debug=True)
