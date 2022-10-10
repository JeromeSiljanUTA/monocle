from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

message = 'ollo everybody'
image_path = ''

@app.route('/')
def hello_world():
    return render_template('index.html', message = message)

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        print(uploaded_file.filename)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug = True, port=5001)
    
