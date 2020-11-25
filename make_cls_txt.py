fp = open('train.txt', "r")
fa = open('newtrain.txt', "a")
count = 0
while (count <809):
    line = fp.readline()
    class_id = line.split("/")[6].split("_")[0]
    print(line)
    if(int(class_id) == 0):
        for i in range(3):
            fa.write(line)    
    fa.write(line)
    count += 1


