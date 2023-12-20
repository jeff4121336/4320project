from flask import Flask, render_template, request
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/audio')
def check():
    dir = os.listdir('./audio')
    if len(dir) != 0:
        return '1'
    else:
        return '0'

# route for getting the user uploaded files
@app.route('/upload', methods=['POST'])
def upload():
    fileitem = request.files['file']

    if fileitem:
        upload_dir = './audio'
        file_path = os.path.join(upload_dir, fileitem.filename)
        fileitem.save(file_path)
        return render_template("home.html")

    return 'File uploaded errors'

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=3000, debug=True)