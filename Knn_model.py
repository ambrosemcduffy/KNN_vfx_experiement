import matplotlib.pyplot as plt
from image_segment_study import Knn_test
import cv2
# Passing in the image path to obtain image
kt = Knn_test("data/endgame.jpg")
img = kt.getImages()
# Preprocessing the image to be passed into network
flatten_img = kt.processImages(img)
# Passing in the flatten image to KNN network
segmented_data,labels_reshape,centers = kt.knn_train(flatten_img,k=6, epoch= 5)
# Showing the results..
def showResults(index = 0):
      '''
      this function shows the results of KNN training. 
      '''
      # Obtaining the mask
      mask = kt.getMask(img,labels_reshape,index,True)
      # setting up plotting
      f, (ax1,ax2,ax3) = plt.subplots(1,3, figsize = (20,20))
      ax1.title.set_text("Original Image")
      ax1.imshow(img)
      ax2.title.set_text("Segmented Data")
      ax2.imshow(segmented_data)
      ax3.title.set_text("Masked Image")
      ax3.imshow(mask)
      # showing the image
      plt.show()
showResults(3)