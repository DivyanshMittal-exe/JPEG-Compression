import cv2
import numpy as np

def convolute(img,kernel,type):
    row,col,c = img.shape
    output = img.copy().astype(np.float32)
    for i in range(type,row-type):
        for j in range(type,col-type):
            for k in range(c):
                window = img[i-type:i+type+1,j-type:j+type+1,k]
                output[i,j,k] = np.sum(np.multiply(window.astype(np.float32),kernel.astype(np.float32)))
    return output   

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def gaussian_blur(img):
    
    kernel = np.array([[0,0,1,2,1,0,0],
                       [0,3,13,22,13,3,0],
                       [1,13,59,97,59,13,1],
                       [2,22,97,159,97,22,2],
                       [1,13,59,97,59,13,1],
                       [0,3,13,22,13,3,0],
                       [0,0,1,2,1,0,0]])/1003
    

    return convolute(img,kernel,3)            
        
def sobel_filter(img):
    Gy = np.array([[1,2,1],
                    [0,0,0],
                    [-1,-2,-1]])
    Gx = np.array([[1,0,-1],
                    [2,0,-2],
                    [1,0,-1]])
    
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw_img = bw_img.reshape((bw_img.shape[0],bw_img.shape[1],1)).astype(np.float32)
    output = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    X_edge = convolute(bw_img,Gx,1).astype(np.float32)
    Y_edge = convolute(bw_img,Gy,1).astype(np.float32)
    V = np.sqrt(np.add(np.square(X_edge),np.square(Y_edge)))
    H = np.arctan(np.divide(Y_edge,X_edge,out=np.zeros_like(Y_edge), where=X_edge!=0))
    output[:,:,0] = (NormalizeData(H[:,:,0])*360)
    output[:,:,1] = NormalizeData( V[:,:,0])*200
    output[:,:,2] = (NormalizeData( V[:,:,0])*200)
    output =  cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    return output
    
    

def edge_detect(path,name):
        
        img = cv2.imread(path)
        blured = gaussian_blur(img)
        sobel_img = sobel_filter(blured)
        cv2.imwrite(name, sobel_img)
        cv2.imshow("EdgeDetected", cv2.imread(name))
        cv2.waitKey()


if __name__ == '__main__':
    path = "flower.jpg"
    edge_detect(path,"EdgeDetectedOutput.jpg")