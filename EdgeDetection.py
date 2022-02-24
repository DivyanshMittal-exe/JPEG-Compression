import cv2
import numpy as np

def convolute(img,kernel):
    row,col,c = img.shape
    output = img.copy()
    for i in range(1,row-1):
        for j in range(1,col-1):
            for k in range(c):
                window = img[i-1:i+2,j-1:j+2,k]
                output[i,j,k] = np.sum(np.multiply(window,kernel))
    return output   


def gaussian_blur(img):
    kernel = np.array([[1/16,2/16,1/16],
                       [2/16,4/16,2/16],
                       [1/16,2/16,1/16]])

    return convolute(img,kernel)            
        
def sobel_filter(img):
    Gy = np.array([[1,2,1],
                    [0,0,0],
                    [-1,-2,-1]])
    Gx = np.array([[1,0,-1],
                    [2,0,-2],
                    [1,0,-1]])
    
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw_img = bw_img.reshape((bw_img.shape[0],bw_img.shape[1],1))
    output = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print(output.shape)
    X_edge = convolute(bw_img,Gx)
    Y_edge = convolute(bw_img,Gy)
    V = np.sqrt(np.add(np.square(X_edge),np.square(Y_edge)))
    return np.array([V[:,:,0],V[:,:,0],V[:,:,0]])
    H = np.arctan(np.divide(Y_edge,X_edge))*229.18 + 114.59
    print(V.shape)
    print(H.shape)
    output[:,:,0] = H[:,:,0]
    # output[:,:,1] = V[:,:,0]
    output[:,:,2] = V[:,:,0]
    return cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    
    

def edge_detect(path):
        
        img = cv2.imread(path)
        row,col,c = img.shape
        blured = gaussian_blur(img)
        sobel_img = sobel_filter(blured)

        
        cv2.imshow('Raw',sobel_img)
        cv2.waitKey(0)
        

if __name__ == '__main__':
    path = "flower.jpg"
    edge_detect(path)