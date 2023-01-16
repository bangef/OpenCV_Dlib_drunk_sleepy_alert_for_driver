import cv2
import dlib
import imutils
from imutils.video import VideoStream
from imutils import face_utils
from imutils.video import FPS
from threading import Thread
import time
from scipy.spatial import distance as dist
import playsound
from env import *

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
 
    ear = (A + B) / (2.0 * C)
    return ear

def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)


face_detector = dlib.get_frontal_face_detector()

face_pose_predictor = dlib.shape_predictor('assets/shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

vs = VideoStream(src=0).start()
time.sleep(1.0)

fps = FPS().start()
fps

while True:
    
    fps.update()
    frame = vs.read()
    frame = imutils.resize(frame, width=frame_width)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detected_faces = face_detector(gray, 0)
    
    fps.stop()
    
    fpscor=(frame.shape[1]-110,20)
    cv2.putText(frame, "FPS: {:.2f}".format(fps.fps()), fpscor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

    for rect in detected_faces:
        detected_faces = face_pose_predictor(gray, rect)
        detected_faces = face_utils.shape_to_np(detected_faces)

        leftEye = detected_faces[lStart:lEnd]
        rightEye = detected_faces[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)        
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    t = Thread(target=sound_alarm,args=(alarm_path,))
                    t.deamon = True
                    t.start()

                alertcor=(int(frame.shape[1]/2-180),70)

                cv2.putText(frame, "TRIGGER DROWSINESS ALARM!", alertcor, cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2)

        else:
            COUNTER = 0
            ALARM_ON = False
       
        EARcor=(int(frame.shape[1]/2-150),25)
        cv2.putText(frame, "Eye Aspect Ratio: {:.2f}".format(ear), EARcor,cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, ), 1)
 
    cv2.imshow("Detector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()