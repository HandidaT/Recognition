import numpy as np

#Converts rgb pixel to yuv pixel
def RGB2YUV( rgb ):
    m = np.array([[ 0.29900, 0.58700,  0.11400],
                 [-0.16874, -0.33126, 0.50000],
                 [ 0.50000, -0.41869, -0.08131]])
    yuv = np.dot(m,rgb)
    yuv[1]+=128.0
    yuv[2]+=128.0
    return yuv
