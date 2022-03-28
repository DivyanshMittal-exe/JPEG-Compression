import cv2
import os
import  numpy as np
from Subsample import *
from DFT import *
from Quantisation import *
# from HuffmanEncode import *

class jpeg:
    Chroma_Scale = 2
    
    def __init__(self):
        pass
    
    def write_orignal(self,img,name,header):
        row,col = img.shape
        np.savetxt(name + ".txt", img, fmt="%d", header=header) 
        return os.path.getsize(name + ".txt")
    
    def write_compress(self,data_list,name,header):
        f = open(name + ".txt", "w")
        f.write(header)
        for dat in data_list:
            npstr = np.array2string(dat.astype(np.int8),max_line_width=10000)[1:-1]
            npstr = " ".join(npstr.split())
            # print(npstr)
            f.write(npstr)
            f.write("\n")
        f.close()
        return os.path.getsize(name + ".txt")
        
    
    def compress_channel(self,img,name,header):
        row,col = img.shape
        data_list = []
        # data_arr = np.empty(shape=(0, 0))
        r,c = row//8,col//8
        for i in range(r):
            for j in range(c):
                block = img[i*8:(i+1)*8,j*8:(j+1)*8]
                block = dft(block)
                block = Quant_Y(block)
                arr = zigzagflat(block)
                arr = np.trim_zeros(arr, 'b')
                data_list.append(arr)
        Yheader = str(r) + " " + str(c) + " " + header + "\n"
        return self.write_compress(data_list,name+header,Yheader)
        
    
    def compress(self,path,un_compress_name,compress_name):
        img = cv2.imread(path)
        row,col,c = img.shape
        
        r = self.write_orignal(img[:,:,0],un_compress_name + "R","R")
        g = self.write_orignal(img[:,:,1],un_compress_name + "G","G")
        b = self.write_orignal(img[:,:,2],un_compress_name + "B","B")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        
        Y  = img[:,:,0]
        Cb  = img[:,:,1]
        Cr  = img[:,:,2]
        
        
        Cb = avg_subsample(Cb,self.Chroma_Scale)
        Cr = avg_subsample(Cr,self.Chroma_Scale)
         
        comp_y = self.compress_channel(Y,compress_name,"Y")
        comp_cb = self.compress_channel(Cb,compress_name,"Cb")
        comp_cr = self.compress_channel(Cr,compress_name,"Cr")
        
        print("Original image size: " + str(r+g+b))
        print("Compressed image size: " + str(comp_y+comp_cb+comp_cr))
        
        print("Compressed size compared to original: " + str(round((comp_y+comp_cb+comp_cr)*100/(r+g+b),3)) + "%")
    
    
    def lines_to_matrix(self,data_list):
        pass
    
    def uncompress(self,y,cb,cr):
        f = open(y + ".txt", "r")
        y = f.readlines()
        f.close()
        
        f = open(cb + ".txt", "r")
        cb = f.readlines()
        f.close()
        
        f = open(cr + ".txt", "r")
        cr = f.readlines()
        f.close()
        
        self.lines_to_matrix(y)

if __name__ == '__main__':
    a = jpeg()
    path = "flower.jpg"
    # path = "dog.NEF"
    a.compress(path,"original","compressed")