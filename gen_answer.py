
rd_path = r'D:\Mammograph\DDSM_training_dataset/test.txt'
rd_path_id = r'D:\Mammograph\mammoDDSM/'
wb_path = r'D:\Mammograph\DDSM_training_dataset/ans.txt'


fp = open(rd_path, 'r')
for i in range(314): # 151 lines
    patient_id = "P_" + fp.readline().split("P_")[1].split('\n')[0]
    get_ans_path = rd_path_id + patient_id.split('.jpg')[0] + '.txt'
    f = open(get_ans_path , "r")
    ans = f.readline().split(" ")[0]
    fa = open(wb_path, "a")
    fa.write(ans + '\n')