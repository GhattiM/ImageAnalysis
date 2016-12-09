from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from imutils import paths
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os
def extract_color_histogram(image, bins=(8, 8, 8)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
 
	# handling the normalization of histogram
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
 
	# else, performing "in place" normalization
	else:
		cv2.normalize(hist, hist)
 
	return hist.flatten()
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))
 
# initialize the data matrix and labels list
data = []
labels = []
for (i, imagePath) in enumerate(imagePaths):
	
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	hist = extract_color_histogram(image)
	data.append(hist)
	labels.append(label)
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))
le = LabelEncoder()
labels = le.fit_transform(labels)
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	np.array(data), labels, test_size=0.25, random_state=42)

# train the linear regression clasifier
print("[INFO] training Linear SVM classifier...")
model = LinearSVC()
model.fit(trainData, trainLabels)

# evaluate the classifier
print("[INFO] evaluating classifier...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions,
	target_names=le.classes_))
h=.02

#Plotting the Images data
X = trainData[:, :2]+ np.r_[np.random.randn(1853, 2) - [2, 2], np.random.randn(1853, 2) + [2, 2]]
y = testLabels
clf = SVC(kernel='linear')
clf.fit(X,y)
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-4,4)
yy = a * xx - clf.intercept_[0] / w[1]
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])
h0 = plt.plot(xx, yy, 'k-', label="SVM div line")
plt.xlabel('Images_Data')
plt.ylabel('Predicted_Labels')
plt.title('SVM on NASA Images')
plt.plot(xx, yy_down, 'k--',c ='b',label = "mars")
plt.plot(xx, yy_up, 'k--',c='r',label = "marsrover")
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, facecolors='none')
plt.scatter(X[:, 0],X[:, 1],c = y,cmap=plt.cm.Paired)
plt.axis('tight')
plt.legend()
plt.show()

