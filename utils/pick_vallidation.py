"""
        pick validation set for training CBIS-DDSM MLP
        Usage : use with pick.txt
        Args:

        Returns:
"""
rd = r'G:\project_yolo\14_1024_DDSM_original_anchor\training_dataset/train.txt'
rd_pick = r'G:\project_yolo\14_1024_DDSM_original_anchor\training_dataset/pick.txt'
wb = r'G:\project_yolo\14_1024_DDSM_original_anchor\training_dataset/train1297.txt'
fp_pick = open(rd_pick, "r")
pick_list = []
while (1):
    line = fp_pick.readline().split('\n')[0]
    pick_list.append(line)
    if line =="":
        break
    else: continue

fp = open(rd, "r")
fa = open('test1297.txt', "a")
i=0
while (1):
    line = fp.readline().split('/')[1].split('.')[0].split('\n')[0]
    if line =="":
        break
    elif line not in pick_list:
        context = 'G:\project_yolo\project_mammo/training_dataset\images/' + line + '.jpg'
        fa.write(context+ "\n")
        i += 1
print(i)
