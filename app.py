# Student ID: 1155158477
# Name: YAU YUK TUNG

from pydub import AudioSegment
import os
import speech_recognition as sr
from flask import Flask, flash, request, redirect, render_template, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup 
import requests 
from googletrans import Translator, constants
from pprint import pprint
from translate import Translator
from gtts import gTTS

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")

ALLOWED_EXTENSIONS = {'wav'}
path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')
OUTPUT_FOLDER = os.path.join(path, 'output')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

if not os.path.isdir(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

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
        returnText = audioToText(filename)

        translator = Translator(to_lang="zh")
        if returnText == "":
            translatedText = "Translation failed"
        else:
            translation = translator.translate(returnText)
            translatedText = translation if translation is not None else "Translation failed"
            
            for file_name in os.listdir(OUTPUT_FOLDER):
                file_path = os.path.join(OUTPUT_FOLDER, file_name)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)

            file_path = os.path.join(OUTPUT_FOLDER, filename)
            tts = gTTS(translatedText, lang='zh')
            tts.save(file_path)

    return render_template("home.html", text=returnText, originalAudioLink="http://127.0.0.1:3000/uploads/" + filename,
                            translatedAudioLink="http://127.0.0.1:3000/output/" + filename, translatedText=translatedText)

def audioToText(filename):
    recognizer = sr.Recognizer()
    audio = UPLOAD_FOLDER + '/' + filename

    with sr.AudioFile(audio) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language="en-US")
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            print("Sorry, there was an error processing your request.")
            return "Sorry, there was an error processing your request."

# e.g. http://127.0.0.1:3000/uploads/OSR_us_000_0018_8k.wav
@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output/<filename>')
def output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=3000, debug=True)