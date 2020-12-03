import csv
# import xlsxwriter

fp = open('newtrain_1.txt', "r")
count = 0
path = r'D:\Mammograph\training_dataset\JPEGImages'

dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,'G': 6, 'H': 7, 'I': 8, 'J': 9 }

def str2num(img_name):
    new_str = ''
    for name_str in img_name:
        name_str = dict[name_str]
        new_str = new_str + str(name_str)

    return new_str

# workbook = xlsxwriter.Workbook(r'newtrain_1.xlsx')
# worksheet = workbook.add_worksheet()
# row = 0
# column = 0
count = 0
fa = open('gen_newtrain.txt', "a")
while (count <1223):
    line = fp.readline()
    tail = line.split("/")[6].split("_")[2]
    encode_part = line.split("/")[6].split("_")[1]
    new_str = str2num(encode_part)
    train_line = path + '\Case_' + new_str + '_' + tail
    fa.write(train_line)
    # print(train_line)
    count += 1
    # worksheet.write(row, column, train_line)
    # row += 1
# workbook.close()
print(count)