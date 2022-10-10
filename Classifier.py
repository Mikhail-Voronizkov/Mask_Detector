import numpy as np
import cv2
from keras.models import load_model

class Inference():
    def __init__(self):
        try:
            self.model = load_model('saved_model')
        except Exception as e:
            print("Exception", e)
            
    def preprocess(self, img, box):
        face_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        face_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        face_img = cv2.resize(face_img, (100,100))
        face_img = face_img / 255.0
        face_img = np.reshape(face_img, (1, 100, 100, 1))
    
        return face_img
            
    def run_inference(self, img, box):  
        label = None
        
        try:
            pre_img = self.preprocess(img, box)
            pred = self.model.predict(pre_img)
            label = np.argmax(pred, axis=1)[0]
        except Exception as e:
            print("Exception", e)
            
        return label