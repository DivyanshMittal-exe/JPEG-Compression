import cv2
import  numpy as np
from Subsample import *

class jpeg:
    Chroma_Scale = 2
    
    def __init__(self):
        pass
    
    
    def compress(self,path):
        
        img = cv2.imread(path)
        row,col,c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        
        Y  = img[:,:,0]
        Cb  = img[:,:,1]
        Cr  = img[:,:,2]
        Cb = avg_subsample(Cb,self.Chroma_Scale)
        Cr = avg_subsample(Cr,self.Chroma_Scale)
        
        
        cv2.imshow('Raw',Cb)
        cv2.waitKey(0)
        pass

if __name__ == '__main__':
    a = jpeg()
    path = "lake.png"
    # path = "dog.NEF"
    a.compress(path)