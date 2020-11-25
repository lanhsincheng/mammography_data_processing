rd_path = r'G:\project_yolo\project_mammo\training_dataset/train-gen-test.txt'
rd_annotation_path = r'G:\project_yolo\project_mammo\training_dataset\images/'
wb_path = r'G:\project_yolo\project_mammo\training_dataset/train-gen-mal.txt'


fp = open(rd_path, "r")
fa = open(wb_path, "a")

def fetch_label(filename):
    fetch_annotation_pth = rd_annotation_path + filename + '.txt'
    ff = open(fetch_annotation_pth, "r")
    line = ff.readline().split(' ')[0]

    return line

count = 0
while (1):
    line = fp.readline()
    if line == "":
        break
    else:
        filename = line.split("/")[1].split(".")[0]
        classid = fetch_label(filename)
        if(int(classid) == 0): # malignant
            for i in range(2):
                fa.write(line)
        fa.write(line)
        count += 1
    print(count)


