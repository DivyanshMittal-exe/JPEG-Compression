import cv2
import  numpy as np
from Subsample import *
from DFT import *
from Quantisation import *
from HuffmanEncode import *

class jpeg:
    Chroma_Scale = 2
    
    def __init__(self):
        pass
    
    def write_orignal(img,name,header):
        row,col = img.shape
        np.savetxt(name + ".txt", img, fmt="%d", header=header) 
        
    
    def compress(self,path,un_compress_name,compress_name):
        
        img = cv2.imread(path)
        row,col,c = img.shape
        
        self.write_orignal(img[:,:,0],un_compress_name + "R","R")
        self.write_orignal(img[:,:,1],un_compress_name + "G","G")
        self.write_orignal(img[:,:,2],un_compress_name + "B","B")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        
        Y  = img[:,:,0]
        Cb  = img[:,:,1]
        Cr  = img[:,:,2]
        
        
        Cb = avg_subsample(Cb,self.Chroma_Scale)
        Cr = avg_subsample(Cr,self.Chroma_Scale)
         
        data_list = []
        # data_arr = np.empty(shape=(0, 0))
        r,c = row//8,col//8
        for i in range(r):
            for j in range(c):
                block = Y[i*8:(i+1)*8,j*8:(j+1)*8]
                block = dft(block)
                block = Quant_Y(block)
                arr = zigzagflat(block)
                arr = np.trim_zeros(arr, 'b')
                data_list.append(arr)
                
        self.write_compress()
        # data_arr = np.append(data_arr,arr)
        # print(data_arr.shape)
        # codecY,hof_dataY = hof(data_arr)
        # print(len(codecY.decode(hof_dataY)))
        # print(codecY)
        # print(len(hof_dataY))
                
        
        cv2.imshow('Raw',Y)
        cv2.waitKey(0)
        pass

if __name__ == '__main__':
    a = jpeg()
    path = "mountain.jpg"
    # path = "dog.NEF"
    a.compress(path)