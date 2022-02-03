import numpy as np
from dahuffman import HuffmanCodec

buff = np.array([[0,  1,  5,  6,  14, 15, 27, 28],
                [2,  4,  7,  13, 16, 26, 29, 42],
                [3,  8,  12, 17, 25, 30, 41, 43],
                [9,  11, 18, 24, 31, 40, 44,53],
                [10, 19, 23, 32, 39, 45, 52,54],
                [20, 22, 33, 38, 46, 51, 55,60],
                [21, 34, 37, 47, 50, 56, 59,61],
                [35, 36, 48, 49, 57, 58, 62,63]])

flat = np.argsort(buff.flatten())

def zigzagflat(block):
    return block.flatten()[flat]

def zigzagbuff(arr):
    return arr[buff].reshape([8,8])

def hof(block):
    data = zigzagflat(block)
    vals, freq = np.unique(data, return_counts=True)
    codec_dict = dict(zip(vals, freq))
    print(codec_dict)
    codec = HuffmanCodec.from_frequencies(codec_dict)
    
    print(codec.decode(codec.encode(data)))
    return codec.encode(data)

if __name__ == '__main__':
    from DFT import *
    from Quantisation import *
    test = np.array([[154,123,123,123,123,123,123,136],
                [ 192,180,136,154,154,154,136,110],
                [ 254,198,154,154,180,154,123,123],
                [ 239,180,136,180,180,166,123,123],
                [ 180,154,136,167,166,149,136,136],
                [ 128,136,123,136,154,180,198,154],
                [ 123,105,110,149,136,136,180,166],
                [ 110,136,123,123,123,136,154,136]])
    
    print(hof(Quant_Y(dft(test))))
    # print(zigzagbuff(zigzagflat(Quant_Y(dft(test)))))