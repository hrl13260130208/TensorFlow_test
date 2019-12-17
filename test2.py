

import cv2
import os

# file=r"C:\Users\zhaozhijie.CNPIEC\Desktop\es\02视频\2018.06.02-09.59.14elsearch.mp4"
# path=r"C:\data\png"

if __name__ == '__main__':
    d={}
    with open(r"C:\data\new_temp_subject_file.txt",encoding="utf-8") as f:
        for line in f.readlines():
            line=line.replace("\n","")
            i=line.split("##")
            d[i[1]]=line

    w=open(r"C:\data\tmp\t.txt","w+",encoding="utf-8")
    for name in os.listdir(r"C:\data\tmp3"):
        if name.replace(".txt","") in d.keys():
            w.write(d[name.replace(".txt","")]+"\n")





    # src = cv2.imread(r'C:\Users\zhaozhijie.CNPIEC\Desktop\th (1).jpg')
    # dst = cv2.blur(src, (10, 10))
    # # dst = cv2.medianBlur(src, 5)
    # # dst = cv2.GaussianBlur(src, (15,15),0)
    # cv2.imshow('dst', dst)
    # cv2.imshow("org",src)
    # cv2.waitKey(0)

    # videoCapture = cv2.VideoCapture(file)
    # i=0
    # while True:
    #     success, frame = videoCapture.read()
    #     i += 1
    #
    #     if not success:
    #         print('video is all read')
    #         break
    #
    #     savedname = os.path.join(path, 'jpg_' + str(i) + '.jpg')
    #     cv2.imwrite(savedname, frame)

        # print('image of %s is saved' % (savedname))




