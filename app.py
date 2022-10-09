from flask import Flask
from werkzeug import secure_filename
import os

# def startup():
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLD = '/Users/blabla/Desktop/kenetelli/htmlfi'
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_FOLD)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(os.path.join(app.config['UPLOAD_FOLDER'], f.filename)))
        return 'file uploaded successfully'

if __name__ == '__main__':
    app.run(debug = True, port=5001)

# startup()
