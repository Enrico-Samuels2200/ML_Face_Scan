import os
import cv2
import pickle
import pymongo
import numpy as np

from datetime import datetime

date_time = datetime.now()
formatted_date = date_time.strftime("%d/%m/%Y")
formatted_time = date_time.strftime("%H:%M")

current_id = 0

signed_in = []
ID = []

base_dir = os.path.dirname(os.path.abspath(__file__))

#  Connects to mongo database.
api = 'mongodb+srv://User2200:eumeLuzLWXqGxcVd@clock-in-system.6tmxo.mongodb.net/user_clock_In?retryWrites=true&w=majority'

client = pymongo.MongoClient(api)
db = client.get_database('user_clock_In')
collection = db.time_table

#  Returns all data from collection time_table.
data = list(collection.find({}))

#  Returns id from database and update ID list with all id's that exist in the database.
def update_id():
    try:
        for user in data:
            signed_in.append(user["employee"])
            ID.append(user["_id"])
        current_id = ID[-1]
    except:
        current_id = 1
    return current_id


#  If the name doesn't exist it'll be added along with a unique id to the database.
def check_sign_in(name):
    global current_id
    try:
        #  Gets the last id in the database collection.
        current_id += 1
        user_data = {"_id": current_id, "employee": name, "date": formatted_date, "time": formatted_time}
        collection.insert_one(user_data)
        signed_in.append(name)
        
    except:
        current_id = 1

def run_scan():
    while True:
        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        #  Draws rectangle around items of interest.
        #  Adds text to notify the user there are changes in the image/video feed.
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (221,160,221), 3)
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            id_, conf = recognizer.predict(roi_gray)

            if conf >= 45 and conf <= 85:
                #  Draws rectangle around items of interest.
                #  Adds text to notify the user there are changes in the image/video feed.
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (30, 255, 255)
                stroke = 2
                cv2.putText(img, name.title(), (x, y+200), font, 0.5, color, stroke, cv2.LINE_AA, False)
                    
                if name not in signed_in:
                    check_sign_in(name)

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #  Teminates the window displaying the video and end the video streaming process.
    cap.release()
    cv2.destroyAllWindows()

#  Get the face cascade file from the root directory of this app.
face_cascade = cv2.CascadeClassifier(os.path.join(base_dir, r"cascades\haarcascade_frontalface_default.xml"))
base_dir = os.path.dirname(os.path.abspath(__file__))

#  Recognizer returns trained data that is used to identify familiar faces.
#  Get the trainner.yml file from the root directory of this app.
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(base_dir, "trainner.yml"))

#  Get the pickle file from the root directory of this app. 
with open(os.path.join(base_dir, "labels.pickle"), "rb") as data_file:
    labels = pickle.load(data_file)
    labels = {v:k for k,v in labels.items()}

#  Set cap parameters to 0 to enable a webcam. Current revieves feed from a ip webcam.
cap = cv2.VideoCapture(r"http://192.168.43.1:8080/video")

#  Runs if file is ran locally
if __name__ == "__main__":
    update_id()
    run_scan()