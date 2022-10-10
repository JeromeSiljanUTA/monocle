from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import cv2
app = Flask(__name__)

upload_location = 'uploads/'

@app.route('/upload')
def upload_file():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
   if request.method == 'POST':
      f = request.files['file']
      f.save(f'{upload_location}{secure_filename(f.filename)}')
      img_path = f'{upload_location}{secure_filename(f.filename)}'
      print(img_path)
      img = cv2.imread(img_path)
      cv2.imshow('uploaded image:', img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
      return 'file uploaded successfully'

if __name__ == '__main__':
    app.run(debug = True, port=5001)
