from pydub import AudioSegment
import os
import speech_recognition as sr
from flask import Flask, flash, request, redirect, render_template, url_for
from werkzeug.utils import secure_filename


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

ALLOWED_EXTENSIONS = {'wav'}
path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def removePreviousFiles():
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST'])
def upload_file():
    removePreviousFiles()
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            audioToText(filename)
            return redirect(request.url)
    return ''

def audioToText(filename):
    recognizer = sr.Recognizer()

    audio = UPLOAD_FOLDER + '/' + filename

    with sr.AudioFile(audio) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language="en-US")
            print(f"You said: {text}")
            return render_template("home.html", text=text, originalAudio=audio)
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return render_template("home.html", text="Sorry, I couldn't understand that.", originalAudio=audio)
        except sr.RequestError:
            print("Sorry, there was an error processing your request.")
            return render_template("home.html", text="Sorry, there was an error processing your request.", originalAudio=audio)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=3000, debug=True)