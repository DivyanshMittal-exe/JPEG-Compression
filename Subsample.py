import numpy as np

def avg_subsample(channel,scale):
        row,col = channel.shape
        channel  = channel[(row%scale//2):row - (row%scale//2),(col%scale//2):col - (col%scale//2)]
        c = np.zeros([row//scale,col//scale])
        row_f,col_f = c.shape
        
        for i in range(row_f-1):
            for j in range(col_f-1):
                c[i][j] = np.mean(channel[i*scale:(i+1)*scale,j*scale:(j+1)*scale])
        return c
    
def max_subsample(channel,scale):
    row,col = channel.shape
    c = np.zeros(row//scale,col//scale)
    row_f,col_f = c.shape
    
    for i in range(row_f-1):
        for j in range(col_f-1):
            c[i][j] = np.amax(channel[i*scale:(i+1)*scale,j*scale:(j+1)*scale])
    return c