
import os
import jieba


path="C:/File/ResultFile/input/tensorflow/game.of.thrones.1080p.bluray"
list=os.listdir("C:/File/ResultFile/input/tensorflow/game.of.thrones.1080p.bluray")
output_file=open("C:/File/ResultFile/input/tensorflow/part-0001","a+",encoding="UTF-8")

for file in list:
    file=open(path+"/"+file,"r",encoding="utf-8")

    while True:
        line = file.readline()
        if line:
            line = line.strip()
            first=line.rfind(",,")
            #print(first)
            if first != -1:
                secend=line.find("\\N{\\fs6}")
                #print(secend)
                if secend !=-1:
                    line=line[first+2:secend]
                    num = line.rfind("}")
                    if num !=-1:
                        line=line[num+1:]

                    seg_list = jieba.cut(line)
                    segments = ""
                    for str in seg_list:
                        segments = segments + " " + str
                    segments = segments + "\n"
                    output_file.write(segments)
        else:
            break


