import os
import shutil
base_dir = "E:/james_frizzell_rao/large_barges_files"
move_dir = "D:/TAMU Work/TAMU 2021 FALL/OCEN 485/Analysis Data/large_barge/"


for subdirs, dirs, files in os.walk(base_dir):
    for file in files:
        curr_path = os.path.join(subdirs, file)
        if curr_path[-3:] == 'LIS':
            curr_num = subdirs.split('/')[2].split('\\')[1]
            print(subdirs)
            new_name = file.split('.')[0]+curr_num+'.LIS'
            shutil.copy(curr_path, move_dir+new_name+'.LIS')

