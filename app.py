from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from model.AI_model import AI
import logging

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/ai/medicine', methods=['POST'])
def getMedicineName():
    if request.method == 'POST':
        directory_path = 'images/'
        medicine_images = request.files.getlist('imageFileList')
        logging.debug(medicine_images)
        merged_file_name = medicine_images[0].filename
        logging.debug(merged_file_name)
        image_name = []
        for image in medicine_images:
            file_name = secure_filename(directory_path + image.filename)
            image_name.append(file_name)
            image.save(file_name)
        print(image_name)
        ai_model = AI()
        ai_model.combination(image_name[0], image_name[1], merged_file_name)
        result = ai_model.test(merged_file_name + ".merged_image.jpg")
        return {'result' : "%s" % result}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
