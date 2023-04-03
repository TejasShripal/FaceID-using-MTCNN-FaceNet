
from numpy import expand_dims


def get_embeddings(embedder, face_pixels):
    face_pixels = face_pixels.astype('float32')
    samples = expand_dims(face_pixels, axis=0)
    embeddings = embedder.embeddings(samples)
    return embeddings