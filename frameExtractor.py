"""
This script extracts frames from videos and convert them into a numpy array
"""

import cv2
from os import listdir
import numpy as np

def getData():
    maxFrames = 40  # maximum frames
    dims = 256  # pixels
    input_folder = 'C:\\Users\\vision\\Downloads\\DATASET\\HockeyFight1\\'
    output_folder = 'D:\\dataset\\'

    output_x = []
    output_y = []
    list = listdir(input_folder)
    vidCount = 0
    for vid in list:
        vidCount += 1
        if vidCount == 5:
            break
        yLabel = -1
        if vid.find('fi') != -1:
            yLabel = 1
        else:
            if vid.find('no') != -1:
                yLabel = 0
        vidObj = cv2.VideoCapture(input_folder + vid)
        # length = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
        framedVideo = []
        count = 0
        success = 1
        while success and count < maxFrames:
            success, image = vidObj.read()
            resize = cv2.resize(image, (dims, dims))
            framedVideo.append(resize)
            # print(image.shape)
            cv2.imwrite(output_folder + "frame%d.jpg" % count, resize)
            count += 1
            
        output_x.append(np.array(framedVideo))
        output_y.append(yLabel)

    output_x = np.array(output_x)
    output_y = np.array(output_y).reshape(-1,1)
    #print(output_x.shape)
    #print(output_y.shape)
    #for i in range(output_y.size):
        #print(output_y[i][0])
    print("Done../")
    return (output_x,output_y)

    """
    for i in range(output_x.shape[2]):
        for j in range(output_x.shape[2]):
            print(output_x[0][0][i][j][0], end=" ")
        print()
    print()

    for i in range(output_x.shape[2]):
        for j in range(output_x.shape[2]):
            print(output_x[0][1][i][j][0], end=" ")
        print()
    print()
    """
