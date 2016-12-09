import os
def rename_mars():
    file_list = os.listdir(r"E:\marsimages\marspic1")
    saved_path = os.getcwd()
    os.chdir(r"E:\marsimages\marspic1")
    for file_name in file_list:
        for i in range(5000):
            os.rename(file_name,"mars."+str(i)+".jpg")
    os.chdir(saved_path)
def rename_marsrover():
    file_list = os.listdir(r"E:\marsimages\marspic2")
    os.chdir(r"E:\marsimages\marspic2")
    for file_name in file_list:
        for i in range(5000):
            os.rename(file_name,"marsrover."+str(i)+".jpg")
    os.chdir(saved_path)
rename_mars()
rename_marsrover()
