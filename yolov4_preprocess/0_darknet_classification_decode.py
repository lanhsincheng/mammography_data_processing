import csv
import xlsxwriter

fp = open('newtrain_1.txt', "r")
count = 0
path = r'D:\Mammograph\training_dataset\JPEGImages'

def str2num(img_name):
    new_str = ''
    for name_str in img_name:
        name_str = int(name_str) - 65
        new_str = new_str + chr(name_str)

    return new_str

workbook = xlsxwriter.Workbook(r'newtrain_1.xlsx')
worksheet = workbook.add_worksheet()
row = 0
column = 0
while (count <1223):
    line = fp.readline()
    tail = ine.split("/")[6].split("_")[2]
    encode_part = line.split("/")[6].split("_")[1]
    new_str = str2num(encode_part)
    train_line = path + '\Case_' + new_str + tail
    worksheet.write(row, column, train_line)
    row += 1
workbook.close()
