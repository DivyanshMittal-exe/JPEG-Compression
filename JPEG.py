import cv2
import os
import  numpy as np
from Subsample import *
from DFT import *
from Quantisation import *

class jpeg:
    Chroma_Scale = 1
    
    def __init__(self,Chroma_Scale):
        self.Chroma_Scale = Chroma_Scale
    
    def write_orignal(self,img,name,header):
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
        r,c = row//8,col//8
        for i in range(r):
            for j in range(c):
                block = img[i*8:(i+1)*8,j*8:(j+1)*8]
                block = dft(block)
                if header != "Y":
                    block = Quant_C(block)
                else:
                    block = Quant_Y(block)
                arr = zigzagflat(block)
                arr = np.trim_zeros(arr, 'b')
                data_list.append(arr)
        Yheader = str(r) + " " + str(c) + " " + header + "\n"
        return self.write_compress(data_list,name+header,Yheader)
        
    def trim(self,img):
        row,col,c = img.shape
        a = self.Chroma_Scale*8
        row_trim = (row - (row//a)*a)
        col_trim = (col - (col//a)*a)
        row_trim_l = (row - (row//a)*a)//2
        col_trim_l = (col - (col//a)*a)//2
        row_trim_r = row_trim - row_trim_l
        col_trim_r = col_trim - col_trim_l
        return img[row_trim_l:row-row_trim_r,col_trim_l:col-col_trim_r,:]

        
    
    def compress(self,path, name):
        img = cv2.imread(path)

        try: 
            img = self.trim(img)
            
            r = self.write_orignal(img[:,:,0], name + "_original_R","R")
            g = self.write_orignal(img[:,:,1], name + "_original_G","G")
            b = self.write_orignal(img[:,:,2], name + "_original_B","B")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            
            Y  = img[:,:,0]
            Cb  = img[:,:,2]
            Cr  = img[:,:,1]
            
            Cb = max_subsample(Cb,self.Chroma_Scale)
            Cr = max_subsample(Cr,self.Chroma_Scale)
            
            comp_y = self.compress_channel(Y, name + "_compressed_","Y")
            comp_cb = self.compress_channel(Cb, name + "_compressed_","Cb")
            comp_cr = self.compress_channel(Cr, name + "_compressed_","Cr")
            
            print("Original image size: " + str(r+g+b))
            print("Compressed image size: " + str(comp_y+comp_cb+comp_cr))
            print("Compressed size compared to original: " + str(round((comp_y+comp_cb+comp_cr)*100/(r+g+b),3)) + "%")
        except:
            print("Compression Failed - Given File Not Found")
    
    def lines_to_matrix(self,data_list,header):
        data_list = [data.split() for data in data_list]
        r,c = data_list[0][0:2]
        r,c = int(r),int(c)
        data_list = data_list[1:]
        data_list = [[int(ent) for ent in data] for data in data_list]

        data_list = [np.array(data) for data in data_list]
        data_list = [np.concatenate((data,np.zeros(64,))) for data in data_list]
        if header != "Y":
            data_list = [inv_dft(inv_Quant_C(zigzagbuff(data[:64]))) for data in data_list]
        else:
            data_list = [inv_dft(inv_Quant_Y(zigzagbuff(data[:64]))) for data in data_list]
            

        return np.concatenate([np.concatenate(data_list[c*i:c*(i+1)], axis=1) for i in range(r)], axis=0)
    
    def uncompress(self, name):
        try:
            f = open( name + "_compressed_Y.txt", "r")
            y = f.readlines()
            f.close()
            
            f = open( name + "_compressed_Cb.txt", "r")
            cb = f.readlines()
            f.close()
            
            f = open( name + "_compressed_Cr.txt", "r")
            cr = f.readlines()
            f.close()
            
            y =  self.lines_to_matrix(y,"Y")
            cb = self.lines_to_matrix(cb,"Cb")
            cb = supersample(cb,self.Chroma_Scale)
            cr = self.lines_to_matrix(cr,"Cr")
            cr = supersample(cr,self.Chroma_Scale)
            img = np.dstack((y, cr, cb)).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
            cv2.imwrite(name + "compressed.png", img)
            cv2.imshow('Compressed Image',img)
            cv2.waitKey(0) # waits until a key is pressed
            cv2.destroyAllWindows()
        except:
            print("Uncompression Failed - Given File Not Found")
        

if __name__ == '__main__':
    a = jpeg(1)
    # print("Enter name of file to be compressed : ")
    # path = str(input()).strip()
    # print("Enter name of destination file : ")
    # name = str(input()).strip()

    path = "fourier.webp"
    name = "fourier"
    # path = "dog.NEF"
    a.compress(path, name)
    a.uncompress(name)