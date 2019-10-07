from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from skimage.io import imread, imshow
from skimage import img_as_float

# Load the parrots.jpg picture.
img=imread('parrots.jpg')

# Convert the image by converting all values to an interval from 0 to 1.
img=img_as_float(img)

# Create the "feature-objects" matrix:
# characterize each pixel with three coordinates - intensity values in RGB space.
X=pd.DataFrame(img.reshape((img.shape[0]*img.shape[1],3)),columns=['R','G','B'])

# Run the K-Means algorithm with parameters init = 'k-means ++' and random_state = 241.
km=KMeans(init='k-means++',random_state=241)
km.fit(X.loc[:,'R':'B'].values)

# After selecting the clusters, try to fill in all the pixels assigned to the same cluster
# in two ways: median and middle color across the cluster.
X['Cluster']=km.labels_
mean=X.groupby('Cluster').mean().values
mean_img=np.array([mean[x] for x in X['Cluster']])
mean_img=mean_img.reshape(img.shape)

median=X.groupby('Cluster').median().values
median_img=np.array([median[x] for x in X['Cluster']])
median_img=median_img.reshape(img.shape)

# Measure the quality of the resulting segmentation using the PSNR metric.
psnr_mean=10*np.log10(1.0/np.mean((img - mean_img) ** 2))
psnr_median=10*np.log10(1.0/np.mean((img - median_img) ** 2))

# Find the minimum number of clusters at which the PSNR value is above 20.
for n in range(1,21):
  km=KMeans(n_clusters=n,init='k-means++',random_state=241)
  km.fit(X.loc[:,'R':'B'].values)

  X['Cluster']=km.labels_
  mean=X.groupby('Cluster').mean().values
  mean_img=np.array([mean[x] for x in X['Cluster']])
  mean_img=mean_img.reshape(img.shape)

  median=X.groupby('Cluster').median().values
  median_img=np.array([median[x] for x in X['Cluster']])
  median_img=median_img.reshape(img.shape)

  psnr_mean=10*np.log10(1.0/np.mean((img - mean_img) ** 2))
  psnr_median=10*np.log10(1.0/np.mean((img - median_img) ** 2))

  if psnr_mean>20 or psnr_median>20:
    answer=n
    break

print(answer)
