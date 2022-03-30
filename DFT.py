import numpy as np
import math
import cmath

# dft_coeff = np.array([[0.35355339, 0.35355339,0.35355339,0.35355339,0.35355339,0.35355339,0.35355339,0.35355339],
#                     [0.49039264,0.41573481,0.27778512,0.09754516,-0.09754516,-0.27778512,-0.41573481,-0.49039264],
#                     [0.46193977,0.19134172,-0.19134172,-0.46193977,-0.46193977,-0.19134172,0.19134172,0.46193977],
#                     [0.41573481,-0.09754516,-0.49039264,-0.27778512,0.27778512,0.49039264,0.09754516,-0.41573481],
#                     [0.35355339,-0.35355339,-0.35355339,0.35355339,0.35355339,-0.35355339,-0.35355339,0.35355339],
#                     [0.27778512,-0.49039264,0.09754516,0.41573481,-0.41573481,-0.09754516,0.49039264,-0.27778512],
#                     [0.19134172,-0.46193977,0.46193977,-0.19134172,-0.19134172,0.46193977,-0.46193977,0.19134172],
#                     [0.09754516,-0.27778512,0.41573481,-0.49039264,0.49039264,-0.41573481,0.27778512,-0.09754516]])




def dft(size):
    dft2d = np.zeros((size,size),dtype=complex)
    for i in range(size):
        for j in range(size):
            dft2d[i,j] = cmath.exp(- 2j * np.pi *i*j/size)
    return dft2d
    
dft_coeff = dft(8)

def dft_mat(size):
    arr = np.zeros([size,size])
    for i in range(size):
        for j in range(size):
            a = 1/np.sqrt(size)
            if i != 0:
                a = np.sqrt(2/size)
                a = a*math.cos(((2*j+1)*i*math.pi)/(2*size))
            arr[i,j] = a
    return arr


def dft(block):
    global dft_coeff
    block = block - 128
    if block.shape[0] !=8:
        dft_coeff = dft_mat(block.shape[0])
    
    return np.dot(dft_coeff,block)
    

def inv_dft(block):
    global dft_coeff
    if block.shape[0] !=8:
        dft_coeff = dft_mat(block.shape[0])
    
    mat =  np.dot(np.linalg.inv(dft_coeff),block)
    
    return np.round(mat) + 128



if __name__ == '__main__':
    test = np.array([[154,123,123,123,123,123,123,136],
                    [ 192,180,136,154,154,154,136,110],
                    [ 254,198,154,154,180,154,123,123],
                    [ 239,180,136,180,180,166,123,123],
                    [ 180,154,136,167,166,149,136,136],
                    [ 128,136,123,136,154,180,198,154],
                    [ 123,105,110,149,136,136,180,166],
                    [ 110,136,123,123,123,136,154,136]])
    
    print(dft(test))
    print(inv_dft(dft(test)))
    
    

