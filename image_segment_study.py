import cv2 
import matplotlib.pyplot as plt
import numpy as np
import os

class Knn_test(object):
      def __init__(self,path):
            self.path= path
      def getImages(self):
            '''
            importing in images, and transformaing into rgb instead of BGR
            '''
            if os.path.exists(self.path):
                  img = cv2.imread(self.path)
                  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                  return img
            else:
                  print("path none existent")
                  return None
      def processImages(self,images):
            flatten = images.reshape((-1,3))
            return np.float32(flatten)
      def knn_train(self,flatten,epoch=10,k=2,episilon=1.0, show = False):
            '''
            training the KNN network.. then exporting out the segmented labeled data
            '''
            img = self.getImages()
            # setting the criteria, and training the network
            criteria = (cv2.TermCriteria_EPS+cv2.TERM_CRITERIA_MAX_ITER,epoch,episilon)
            # obtaining the labels
            retval,labels,centers = cv2.kmeans(flatten,k,None,criteria,epoch, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            # obtainig the segmented data
            segmented_data = centers[labels.flatten()]
            # reshaping the data to be viewed, and used for masking
            segmented_data = segmented_data.reshape(img.shape)
            labels_reshape = labels.reshape(img.shape[0], img.shape[1])
            if show:
                  plt.imshow(segmented_data, "gray")
            return segmented_data, labels_reshape
      def getMask(self,img,labels_reshape, cluster = 0, show=False):
            '''
            this function creates a mask.
            '''
            mask_image = np.copy(img)
            mask_image[labels_reshape ==cluster] = [0,0,0]
            if show:
                  plt.figure(figsize=(10,20))
                  plt.imshow(mask_image, cmap="gray")
            return mask_image
            

kt = Knn_test("data/before_got.jpg")
img = kt.getImages()
flatten_img = kt.processImages(img)
segmented_data,labels_reshape = kt.knn_train(flatten_img,k=5, show=True)
mask = kt.getMask(img,labels_reshape,3,True)
   