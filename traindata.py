import utils
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
import os

embedder = FaceNet()
os.chdir('SVM_MTCNN/')
print('LOADING DATABASE')
trainX, trainy = utils.load_dataset('database/')

trainX = utils.newtrain(embedder,trainX)
a, b, c = trainX.shape
trainX = trainX.reshape(a, c)
print(trainX.shape)
print(trainy.shape)
print("reshaped!")
utils.em_compression(trainX, trainy)

#-------model------
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)

out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)


model = SVC(kernel='linear', probability = True)
model.fit(trainX, trainy)
pickle.dump(model, open('model.pkl', 'wb'))






    
    
    