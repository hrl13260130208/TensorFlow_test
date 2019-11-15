
input="C:/File/ResultFile/input/tensorflow/organization/导出.tsv"
out="C:/File/ResultFile/input/tensorflow/organization/input"

file=open(input,encoding="utf-8")
output=open(out,"a+",encoding="utf-8")

def main():
    for line in file.readlines():
        colmns=line.split("\"")
        print(colmns[1])
        output.write(colmns[1]+"\n")

main()