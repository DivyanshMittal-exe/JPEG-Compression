import numpy as np

def avg_subsample(channel,scale):
        row,col = channel.shape
        channel  = channel[(row%scale//2):row - (row%scale//2),(col%scale//2):col - (col%scale//2)]
        c = np.zeros([row//scale,col//scale])
        row_f,col_f = c.shape
        for i in range(row_f):
            for j in range(col_f):
                c[i,j] = np.mean(channel[i*scale:(i+1)*scale,j*scale:(j+1)*scale])
                # print(c[i,j])
                
        return c
    
def max_subsample(channel,scale):
    row,col = channel.shape
    c = np.zeros([row//scale,col//scale])
    
    row_f,col_f = c.shape
    
    for i in range(row_f):
        for j in range(col_f):
            c[i][j] = np.amax(channel[i*scale:(i+1)*scale,j*scale:(j+1)*scale])
    return c

def supersample(channel,scale):
    row,col = channel.shape
    c = np.zeros([row*scale,col*scale])
    for i in range(row):
        for j in range(col):
            c[i*scale:(i+1)*scale,j*scale:(j+1)*scale].fill(channel[i,j])
    
    return c


    

if __name__ == '__main__':
    test = np.array([[154,123,123,123,123,123,123,136],
                    [ 192,180,136,154,154,154,136,110],
                    [ 254,198,154,154,180,154,123,123],
                    [ 239,180,136,180,180,166,123,123],
                    [ 180,154,136,167,166,149,136,136],
                    [ 128,136,123,136,154,180,198,154],
                    [ 123,105,110,149,136,136,180,166],
                    [ 110,136,123,123,123,136,154,136]])
    print(avg_subsample(test,2))
    print(max_subsample(test,2))