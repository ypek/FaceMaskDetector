import urllib
import numpy as np
import cv2
import os

for file_type in ['negatives']:
    for img in os.listdir(file_type):
        line = file_type+'/'+img+'\n'
        with open('negatives.txt','a') as f:
            f.write(line)
