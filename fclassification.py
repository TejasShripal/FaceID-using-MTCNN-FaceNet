import cv2
from mtcnn import MTCNN
from PIL import Image
import numpy as np
from numpy import asarray
from embedding import get_embeddings
from pickle import load
import os
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import time
os.chdir('SVM_MTCNN/')
embedder = FaceNet()
face_detect = MTCNN()
out_encoder = LabelEncoder()
data = np.load('embed.npz')
trainy = data['arr_1']
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
def face_box(img, size = (160,160)):
    pixels = asarray(img)
    results = face_detect.detect_faces(pixels)
    if len(results) == 0:
        return results
    x1,y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(size)
    face_array = asarray(image)
    return get_embeddings(embedder, face_array)

cam = cv2.VideoCapture(0)
model = load(open('model.pkl', 'rb'))
print("Warming up sensors..")
time.sleep(2.0)

while True:
    _, frame = cam.read()
    embed = face_box(frame)
    if len(embed) == 0:
        print("None")
        continue
    label = model.predict(embed)
    prob = model.predict_proba(embed)
    predict_names = out_encoder.inverse_transform(label)
    print(predict_names[0] + str(prob[0,label[0]]))
    if prob[0,label[0]] > 0.8:
        print(predict_names[0] + ', confidence = ' + str(prob[0,label[0]]))
        #frame = cv2.putText(frame, 'classified as ' + predict_names[0] + ': ' + str(prob[0,label[0]]),(50, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),2)
        #cv2.imshow('frame',frame)
    elif label.size == 0:
        print("not detected")
        #cv2.imshow('frame',frame)
    #if cv2.waitKey(1) == ord('q'):
        break
#cam.release()
#cv2.destroyAllWindows()

    
    
    
    
    
    
    