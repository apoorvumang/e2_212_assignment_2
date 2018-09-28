import os
import sys
import cv2
import numpy as np
from sklearn import neighbors
# Create data matrix from a list of images
def createDataMatrix(images):
	numImages = len(images)
	sz = images[0].shape
	data = np.zeros((numImages, sz[0] * sz[1]), dtype=np.float32)
	for i in range(0, numImages):
		image = images[i].flatten()
		data[i,:] = image
	return data

# Read images from a directory
def readImages(path):
	print("Reading images from " + path)
	images = []
	for filePath in sorted(os.listdir(path)):
		for filePath2 in sorted(os.listdir(path + "/" + filePath)):
		    fileExt = os.path.splitext(filePath2)[1]
		    if fileExt in [".jpg", ".jpeg", ".pgm"]:

		      imagePath = path + "/" + filePath + "/" + filePath2
		      im = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2GRAY)
		      if im is None :
		      	print("image:{} not read properly".format(imagePath))
		      else :
			      im = np.float32(im)/255.0
			      images.append(im)

	numImages = len(images)
	if numImages == 0 :
  		sys.exit(0)
	print(str(numImages) + " files read.")
	return images


def readImage(path):
	im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
	if im is None :
	  print("image:{} not read properly".format(imagePath))
	else :
		im = np.float32(im)/255.0
	return im

if __name__ == '__main__':

	# Number of EigenFaces
	NUM_EIGEN_FACES = 320 #number of images total
	# Read images
	images = readImages("images/train")
	testImages = readImages("images/test")
	# Size of images
	sz = images[0].shape
	# Create data matrix for PCA.
	data = createDataMatrix(images)
	# Compute the eigenvectors from the stack of images created
	mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
	averageFace = mean.reshape(sz)
	eigenFaces = []

	lowDimImages = []
	for i in range(len(images)):
		lowDimImages.append(np.dot(eigenVectors, images[i].flatten() - mean.flatten()))

	lowDimTestImages = []
	for i in range(len(testImages)):
		lowDimTestImages.append(np.dot(eigenVectors, testImages[i].flatten() - mean.flatten()))
	# lowDimImages contains the low dimension representations of all
	# train images, same for lowDimTestImages

	for eigenVector in eigenVectors:
		eigenFace = eigenVector.reshape(sz)
		eigenFaces.append(eigenFace)

	idOfTestImage = np.random.randint(0,len(lowDimTestImages))
	testImage = testImages[idOfTestImage]
	lowDimTestImage = lowDimTestImages[idOfTestImage]

	idOfTrainImage = np.random.randint(0,len(lowDimImages))
	trainImage = images[idOfTrainImage]
	lowDimTrainImage = lowDimImages[idOfTrainImage]

	font = cv2.FONT_HERSHEY_SIMPLEX



	valuesOfK = [50,100,200,300]
	trainImageReconConcat = averageFace
	trainImageReconConcat = np.concatenate((trainImageReconConcat, trainImage), axis=1)
	for K in valuesOfK:
		output = averageFace
		for i in range(0, K):
			output = np.add(output, eigenFaces[i] * lowDimTrainImage[i])
		cv2.putText(output,'K='+str(K),(0,20), font, 0.5,(255,255,255))
		trainImageReconConcat = np.concatenate((trainImageReconConcat, output), axis=1)

	testImageReconConcat = averageFace
	testImageReconConcat = np.concatenate((testImageReconConcat, testImage), axis=1)
	for K in valuesOfK:
		output = averageFace
		for i in range(0, K):
			output = np.add(output, eigenFaces[i] * lowDimTestImage[i])
		cv2.putText(output,'K='+str(K),(0,20), font, 0.5,(255,255,255))
		testImageReconConcat = np.concatenate((testImageReconConcat, output), axis=1)
	# Display result at 2x size
	cv2.imshow("Train image reconstruction",cv2.resize(trainImageReconConcat, (0,0), fx=2, fy=2) )
	cv2.imshow("Test image reconstruction", cv2.resize(testImageReconConcat, (0,0), fx=2, fy=2))



	# classification now
	# lowDimImages contains array of image vectors of train images (320 such vectors)
	faceClassifier = neighbors.KNeighborsClassifier(n_neighbors = 3)
	y=[]
	# filling y with target class. every 8 images is 1 class, total 40 classes
	for i in range(0,40):
		faceClass = i
		for j in range(0,8):
			y.append(faceClass)
	faceClassifier.fit(lowDimImages, y)

	#classifier is now trained. we now load test images and their base truth classes

	groundTruth = []
	# filling groundTruth with target class. evergroundTruth 2 images is 1 class, total 40 classes
	for i in range(0,40):
		faceClass = i
		for j in range(0,2):
			groundTruth.append(faceClass)

	prediction = []
	for image in lowDimTestImages:
		prediction.append(faceClassifier.predict([image])[0])

	numCorrect = 0.0
	numWrong = 0.0
	for i in range(0, len(groundTruth)):
		if(groundTruth[i] == prediction[i]):
			numCorrect += 1
		else:
			numWrong += 1
	predictionAccuracy = 100.0*numCorrect/(numCorrect+numWrong)
	print("Correct predictions = " + str(numCorrect))
	print("Incorrect predictions = " + str(numWrong))
	print("Prediction Accuracy = "+ str(predictionAccuracy))

	cv2.waitKey(0)
	cv2.destroyAllWindows()
