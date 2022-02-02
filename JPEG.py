import cv2
import  numpy as np


class jpeg:
    Chroma_Scale = 2
    
    def __init__(self):
        pass
    
    def avg_subsample(self,channel,scale):
        row,col = channel.shape
        channel  = channel[(row%scale//2):row - (row%scale//2),(col%scale//2):col - (col%scale//2)]
        c = np.zeros([row//scale,col//scale])
        row_f,col_f = c.shape
        
        for i in range(row_f-1):
            for j in range(col_f-1):
                c[i][j] = np.mean(channel[i*scale:(i+1)*scale,j*scale:(j+1)*scale])
        return c
    
    def max_subsample(self,channel,scale):
        row,col = channel.shape
        c = np.zeros(row//scale,col//scale)
        row_f,col_f = c.shape
        
        for i in range(row_f-1):
            for j in range(col_f-1):
                c[i][j] = np.amax(channel[i*scale:(i+1)*scale,j*scale:(j+1)*scale])
        return c
    
    def compress(self,path):
        
        img = cv2.imread(path)
        row,col,c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        
        Y  = img[:,:,0]
        Cb  = img[:,:,1]
        Cr  = img[:,:,2]
        Cb = self.avg_subsample(Cb,self.Chroma_Scale)
        Cr = self.avg_subsample(Cr,self.Chroma_Scale)
        
        
        cv2.imshow('Raw',Cb)
        cv2.waitKey(0)
        pass

if __name__ == '__main__':
    a = jpeg()
    path = "lake.png"
    # path = "dog.NEF"
    a.compress(path)