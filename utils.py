from PIL import Image
from os.path import isdir
from os import listdir
from numpy import asarray
from numpy import savez_compressed
from mtcn import face_detect
from numpy import load
from embedding import get_embeddings


#image = cv2.imread('ben.jpg')
def face_box(path, size = (160, 160)):
    img = Image.open(path)
    pixels = asarray(img)
    results = face_detect.detect_faces(pixels)
    x1,y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(size)
    face_array = asarray(image)
    return face_array

def load_faces(directory):
  faces = list()
  for filename in listdir(directory):
    path = directory + filename
    print(path)
    try:
      face = face_box(path)
      faces.append(face)
    except:
      print(path)
      continue
  return faces

def load_dataset(directory):
  X, y = list(), list()
  for subdir in listdir(directory):
    path = directory + subdir + '/'
    print(path)
    if not isdir(path):
        continue   
    faces = load_faces(path)
    labels = [subdir for _ in range(len(faces))]
    print('>loaded %d examples for class: %s' % (len(faces), subdir))
    X.extend(faces)
    y.extend(labels)
    print("arrays and labels generated, passing to compression and decompression!")
  return compression(asarray(X),asarray(y))

def compression(trainX, trainy):
  savez_compressed('data.npz', trainX, trainy)
  print('COMPRESSED!')
  return decomp('data.npz')

def em_compression(trainX, trainy):
  savez_compressed('embed.npz', trainX, trainy)
  
def decomp(fn):
  data = load(fn)
  trainX, trainy = data['arr_0'], data['arr_1']
  print("DECOMPRESSED!!")
  return trainX, trainy
  
def newtrain(embedder, trainX):
  newTrainX = list()
  for face_pixels in trainX:
    embedding = get_embeddings(embedder, face_pixels)
    newTrainX.append(embedding)
  newTrainX=asarray(newTrainX)
  print("Embeddings generated for the dataset!!")
  return newTrainX



    
