import os
from PIL import Image
import numpy as np
import _pickle,gzip
import rgb2yuv

#print(os.listdir('uppercase'))
print(len(os.listdir('uppercase')))
print(os.listdir('uppercase')[0])
a=os.listdir('uppercase')[0].split('.')
print(a[-1][0])

def vectorized_results(label):
    e=np.zeros((62,1))
    e[label]=1.0
    return e

def unicode_tolabel(char):
    uni_code = ord(char)
    if uni_code >= 97:
       return (uni_code - 61)
    else:
       return (uni_code - 55)

def toyuv(list_of_pixels):
      extendedlist=[]#have to extend the nested lists
      for i in list_of_pixels:
         temp=RGB2YUV(np.asarray(i))
         extendedlist.extend(temp)
      return extendedlist

def preparesample(path,filename):
      img=0
      f=filename.split('.')
      print(f)
      img = Image.open(path+"/"+filename)
      if f[-1][0] == 'p':
         img=img.convert('RGB')
      img1=img.resize((28,28))
      width, height = img1.size
      sequence_of_pixels=img1.getdata()
      list_of_pixels = list(sequence_of_pixels)
      data= (np.asarray(toyuv(list_of_pixels)),       vectorized_results( unicode_tolabel(f[0][0])  )  )
      print(type(data),len(data))
      print(type(data[0]),len(data[0]), data[1])
      return data
      
def prepare_data(data,path):
   #datalist=[]
   for i in os.listdir(path):
      #print(i) 
      data.append(preparesample(path,i))


datalist=[]
prepare_data(datalist,'uppercase')














