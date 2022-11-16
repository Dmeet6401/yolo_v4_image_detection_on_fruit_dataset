import glob
import os
import numpy as np
import sys
current_dir = "C:\\Users\\admin\\Desktop\\DX codes\\computer vision\\Drone_obj_det\\Database1\\Database1\\test"
split_pct = 10
# file_train = open("train.txt", "w")
file_val = open("test.txt", "w")
counter = 1
index_test = round(100 / split_pct)
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpeg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        if counter == index_test:
                counter = 1
                file_val.write(current_dir + "\\" + title + '.jpeg' + "\n")
        else:
                file_val.write(current_dir + "\\" + title + '.jpeg' + "\n")
                counter = counter + 1
# file_train.close()
file_val.close()