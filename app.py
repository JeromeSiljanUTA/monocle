from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import cv2
app = Flask(__name__)

upload_location = 'static/'

@app.route('/upload', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        img_path = f'{upload_location}{secure_filename(f.filename)}'
        f.save(img_path)
        return render_template('upload.html', img=img_path)
    else:
        return render_template('upload.html', img='')

if __name__ == '__main__':
    app.run(debug = True, port=5001)
