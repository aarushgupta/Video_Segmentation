
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.cluster import KMeans
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


# In[2]:


def adjacency_matrix(a, nframes):
    b = np.zeros((nframes, nframes), dtype=np.float64)
    for i in range(nframes):
        for j in range(i+1, nframes):
            c = np.sum(np.square(np.subtract(a[i], a[j]), dtype = np.float64))**0.5
            b[i][j] = c
            b[j][i] = c
    return b


# In[4]:


k = 11


# In[5]:


def get_features(nframes, ext='.jpg', path = './Frames'):
     model = VGG16(weights='imagenet', include_top=False)
     feature_list = []
     for i in range(nframes):
         img_path = path+'/frame'+str(i)+ext
         img = image.load_img(img_path, target_size=(224, 224))
         x = image.img_to_array(img)
         x = np.expand_dims(x, axis=0)
         x = preprocess_input(x)
         features = model.predict(x)
         feature_list.append(features[0])
     return np.asarray(feature_list)

# In[8]:


features = get_features(426)


# In[9]:


#print(features.max(), features.min())

b = adjacency_matrix(features, 426)
b = b/200
#b


# In[159]:


deg_mat = np.sum(b, axis=0)


# In[160]:


diag_mat = np.diag(deg_mat)


# In[162]:


laplacian_matrix = diag_mat-b


# In[163]:


#print(laplacian_matrix)


# In[164]:


"""Unnormalized Spectral Clustering"""
#eigen_values, eigen_vectors = np.linalg.eig(laplacian_matrix)
"""Normalized spectral clustering according to Shi and Malik"""
eigen_values, eigen_vectors = np.linalg.eig(np.dot(np.linalg.inv(diag_mat),laplacian_matrix))
"""Normalized spectral clustering according to Ng, Jordan, and Weiss"""
#eigen_values, eigen_vectors = np.linalg.eig(np.dot(np.linalg.inv(diag_mat)**0.5,laplacian_matrix, np.linalg.inv(diag_mat)**0.5))


# In[165]:


# print(eigen_values)
# print(eigen_vectors)


# In[166]:


unnormailzed = eigen_vectors[:,:k]
unnormailzed.shape


# In[168]:


"""For third type of laplacian"""
# factor = np.sqrt((unnormailzed*unnormailzed).sum(axis = 1))
# len(factor)
# for i in range(len(factor)):
#     unnormailzed[i]/factor[i]
# NOT FULLY IMPLEMENTED

# In[169]:


kmeans = KMeans(n_clusters=k).fit(unnormailzed)
# for second type of laplacian eigen_vectors[:,:k]


# In[170]:


y = kmeans.labels_
print(y)


# In[101]:


frames0 = [i for i in range(len(y)) if y[i]==0]
frames1 = [i for i in range(len(y)) if y[i]==1]
frames2 = [i for i in range(len(y)) if y[i]==2]
frames3 = [i for i in range(len(y)) if y[i]==3]
frames4 = [i for i in range(len(y)) if y[i]==4]
frames5 = [i for i in range(len(y)) if y[i]==5]
frames6 = [i for i in range(len(y)) if y[i]==6]
frames7 = [i for i in range(len(y)) if y[i]==7]
frames8 = [i for i in range(len(y)) if y[i]==8]


# In[105]:


print(frames0)
print(frames1)
print(frames2)
print(frames3)
print(frames4)
print(frames5)
print(frames6)
print(frames7)
print(frames8)
