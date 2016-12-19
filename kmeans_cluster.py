import os
import os.path
import numpy as np
import cv2
from matplotlib import pyplot as plt


def dir_list_fn(dir_path):
    dir_ls = []
    cnt = 0
    dir_ls = os.listdir(dir_path)
    for file in dir_ls:
        cnt+=1
    return dir_ls
    
    
def img_arr_fn():
    dr_fl=dir_list_fn("E:\\marsimages\\marspic1\\")
    join_img=[[]]
    for fl in dr_fl:
        img_arr = cv2.imread("E:\\marsimages\\marspic1\\"+fl)
        #img_flt = img_arr.flatten('F')
        #join_img = np.vstack(img_flt)
        join_img = np.vstack(img_arr)
        
    return join_img

result=img_arr_fn()
Z = np.float32(result)
criteria = (cv2.TERM_CRITERIA_EPS, 10,0)
ret,label,center=cv2.kmeans(Z,2,criteria,10,0,flags=2)
print np.shape(result)
A = Z[label.ravel()==0]
B = Z[label.ravel()==1]
plt.scatter(A[:,0],A[:,1],A[:,2])
plt.scatter(B[:,0],B[:,1],B[:,2],c = 'r')
plt.show()
