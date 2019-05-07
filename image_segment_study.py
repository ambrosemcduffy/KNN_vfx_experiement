import cv2 
import matplotlib.pyplot as plt
import numpy as np
# importing in image file
img = cv2.imread("data/ancientone.jpg")
img2 = cv2.imread("data/spidey.jpg")
img2 = cv2.resize(img2,(2624,1240))
data = np.array([img,img2])
# changing the color to RGB from BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# number 
k = 2
def preprocess(img):
      x = img.reshape((-1,3))
      x = np.float32(x)
      return x
flatten_img = preprocess(data)
criteria = (cv2.TermCriteria_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
retval,labels,centers = cv2.kmeans(flatten_img,k,None,criteria,10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshaping 
segmented_data = segmented_data.reshape(data.shape)
labels_reshape = labels.reshape(2,data.shape[1], data.shape[2])
plt.imshow(labels_reshape[1])
def showmask(img,labels_reshape, cluster = 0):
      mask_image = np.copy(img)
      mask_image[labels_reshape[0] ==cluster] = [0,0,0]
      plt.figure(figsize=(10,20))
      plt.imshow(mask_image, cmap="gray")
showmask(data[0].squeeze(), labels_reshape,1)