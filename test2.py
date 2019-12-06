

import cv2
import os

file=r"C:\Users\zhaozhijie.CNPIEC\Desktop\es\02视频\2018.06.02-09.59.14elsearch.mp4"
path=r"C:\data\png"

if __name__ == '__main__':
    videoCapture = cv2.VideoCapture(file)
    i=0
    while True:
        success, frame = videoCapture.read()
        i += 1

        if not success:
            print('video is all read')
            break

        savedname = os.path.join(path, 'jpg_' + str(i) + '.jpg')
        cv2.imwrite(savedname, frame)

        print('image of %s is saved' % (savedname))




