


import os
import glob
import re

def alter_lables(input_txt_paht):
    glob_path = os.path.join(input_txt_path,"*.txt")

    txt_list = glob.glob(glob_path)
    txt_list.sort()
    for txt in txt_list:
        f = open(txt,'r+')
        lines = f.readlines()
        f.close()
        with open(txt,'w+') as file:
            for line in lines:
                oneline1 = line.split(' ')[0]
                # print(line_left)
                if oneline1 == '0':
                    oneline = re.sub(r'0','1',line,1)
                if oneline1 == '1':
                    oneline = re.sub(r'1', '0', line, 1)
                if oneline1 == '2':
                    oneline = re.sub(r'2', '0', line, 1)
                if oneline1 == '3':
                    oneline = re.sub(r'3', '2', line, 1)

                file.write(oneline)
if __name__ == "__main__":
    input_txt_path = "F:/UA-DETRAC/test_txt/"
    alter_lables(input_txt_path)
