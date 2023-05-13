from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import cv2
import test

app = Flask(__name__)

upload_location = "static/"


def return_message(arr_message):
    html_message = ""
    for message in arr_message:
        html_message += message
        html_message += "<br><br>"

    print(html_message)
    return html_message


@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        f = request.files["file"]
        img_path = f"{upload_location}{secure_filename(f.filename)}"
        f.save(img_path)
        return render_template(
            "index.html", img=img_path, message=test.read_img(img_path)
        )
    else:
        return render_template("index.html", img="", message="")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
