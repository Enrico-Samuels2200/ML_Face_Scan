# ML_Face_Scan

# Training the app
To train the application simply add a directory with the desired name in the train directory. Once that is complete activate train.py to generate a pickle file.

# Scanning Faces
To detect a face scan_user.py. By default it'll recieve a video feed through a ip webcam with the use of the android app IP Webcam. To use a built in webcam simply ed cap = cv2.VideoCapture(r'http://192.168.43.1:8080/video')
to cap = cv2.VideoCapture(0).
