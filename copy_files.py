import shutil, os

#script to separate test and train images

#assuming folder "E2 212 Assignment 2_Images" contains all the 40
#face image folders,
#this script divides them into test and train folders
#first 8 images in train, other 2 in test
#same directory structure and naming

if os.path.exists("images"):
    shutil.rmtree("images")
os.mkdir("images")
os.mkdir("images/train")
os.mkdir("images/test")
for i in range(1,41):
    folderName = "E2 212 Assignment 2_Images/s" + str(i)
    os.mkdir("images/train/s" + str(i))
    os.mkdir("images/test/s" + str(i))
    for j in range(1,11):
        fileName = str(j) + ".pgm"
        fullName = folderName + "/" + fileName
        #copy 1 to 8 to folder images/train
        #copy 9 to 10 to folder images/test
        if j <= 8:
            testOrTrain = "train"
        else:
            testOrTrain = "test"
        destinationFolderName = "images/" + testOrTrain + "/s" + str(i) + "/"
        shutil.copy(fullName, destinationFolderName)
