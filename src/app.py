from flask import Flask, render_template, send_from_directory, url_for
import cv2
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from ultralytics import YOLO
import os 

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asldfkjlj'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

photos = UploadSet('photos', IMAGES)

model = YOLO("YOLO_model/best.pt")

def load_img(path):
    print('PATH: ', path)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def yolo_model(model):
    #model = YOLO("YOLO_model") #load yolo model train
    return model

def seg_img(model, img):
    detections = model.predict(img, project="static", name="predictions", save=True)
    #detections = detections[0].plot()

    return detections

configure_uploads(app,photos)
class UploadForm(FlaskForm):
    photo = FileField(

        validators=[
            FileAllowed(photos, 'Only images are allowed'),
            FileRequired('File field should not be empty')
        ]
    )
    submit = SubmitField('Upload')

@app.route('/static/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    detections = None
    static_path = os.path.join(os.getcwd(), 'static')
    # Count folders starting with "predictions" in the static folder
    prediction_folders = [folder for folder in os.listdir(static_path) if folder.startswith('predictions') and os.path.isdir(os.path.join(static_path, folder))]
    count_predictions = len(prediction_folders)
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
        y_model = yolo_model(model)
        image = load_img(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
        detections= seg_img(model=y_model,img=image)
        static_path = os.path.join(os.getcwd(), 'static')
        # Count folders starting with "predictions" in the static folder
        prediction_folders = [folder for folder in os.listdir(static_path) if folder.startswith('predictions') and os.path.isdir(os.path.join(static_path, folder))]
        count_predictions = len(prediction_folders)
    
        #print(detections)
        return render_template('index.html', form=form, file_url=file_url,
                           detections = detections, count_predictions  = count_predictions)
    else:
        file_url = None    
    return render_template('index.html', form=form, file_url=file_url,
                           detections = detections, count_predictions  = count_predictions)
if __name__ == "__main__":
    app.run(debug=True)