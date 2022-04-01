import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import  numpy as np
from matplotlib import pyplot as plt  # Display library
import numpy as np               # Numerical computations

def compress(path):
    img = cv2.imread(path)
    
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
    
    def display_2(im_1, title_1, im_2, title_2):
        """
        Displays two images side by side; typically, an image and its Fourier transform.
        """
        plt.figure(figsize=(12,6))                    # Rectangular blackboard
        plt.subplot(1,2,1) ; plt.title(title_1)       # 1x2 waffle plot, 1st cell
        plt.imshow(im_1, cmap="gray")                 # Auto-equalization
        plt.subplot(1,2,2) ; plt.title(title_2)       # 1x2 waffle plot, 2nd cell
        plt.imshow(im_2, cmap="gray", vmin=-7, vmax=15)  
        plt.show()
        
    def Fourier_threshold(fI, threshold) :
        fI_thresh = fI.copy()                  # Create a copy of the Fourier transform
        fI_thresh[ abs(fI) < threshold ] = 0   # Remove all the small coefficients
        I_thresh = np.real(ifft2(fI_thresh))   # Invert the new transform...

        display_2( I_thresh, "Image",          # And display
                fftshift( np.log(1e-7 + abs(fI_thresh)) ), "Fourier Transform" )
        return fI_thresh
    
    fI = fft2(I)  # Compute the Fourier transform of our slice
    # print(fI)
    display_2( I, "Image", fftshift( np.log(1e-7 + abs(fI)) ), "Fourier Transform" )
    
    fI_n = Fourier_threshold(fI, 70)

    compression = (np.count_nonzero(fI_n))/(np.count_nonzero(fI))*100
    print(compression)
    # Display the logarithm of the amplitutde of Fourier coefficients.
    # The "fftshift" routine allows us to put the zero frequency in
    # the middle of the spectrum, thus centering the right plot as expected.
    # display_2( I, "Image", fftshift( np.log(1e-7 + abs(fI)) ), "Fourier Transform" )
    # img = np.real(np.fft.ifft2(fourier_remover(fourier,100000)))
    # cv2.imshow('Compressed Image',img)
    # cv2.waitKey(0) # waits until a key is pressed
    # cv2.destroyAllWindows()
    


if __name__ == '__main__':
    compress("flower.jpg")