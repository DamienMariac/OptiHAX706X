#%%
# Singular-value decomposition
from numpy import array
from scipy.linalg import svd
from math import sqrt
import numpy as np
import os

# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
#

UU = array([[1/sqrt(2), 1/sqrt(2)], [-1/sqrt(2), 1/sqrt(2)]])
print(UU)

SS = array([[2*sqrt(5), 0], [0, 4*sqrt(5)]])
print(SS)

VVT = array([[-3/sqrt(10), 1/sqrt(10)], [1/sqrt(10), 3/sqrt(10)]])
print(VVT)

AA=np.matmul(np.matmul(UU,SS),VVT)
print(AA)

# SVD
U, s, VT = svd(AA)
print(U-UU)
print(s[1]-SS[0,0])
print(s[0]-SS[1,1])
print(VT-VVT)


#%%
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image

img = Image.open('test.jpeg')
imggray = img.convert('LA')
plt.figure(figsize=(9, 6))
plt.imshow(imggray);

imgmat = np.array(list(imggray.getdata(band=0)), float)
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)
plt.figure(figsize=(9,6))
plt.imshow(imgmat, cmap='gray');

U, sigma, V = np.linalg.svd(imgmat)

reconstimg = np.matrix(U[:, :1]) * np.diag(sigma[:1]) * np.matrix(V[:1, :])
plt.imshow(reconstimg, cmap='gray');

for i in range(10,100,20):
    reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
    plt.imshow(reconstimg, cmap='gray')
    title = "n = %s" % i
    plt.title(title)
    plt.show()

#%%
i1=20
i2=40
sampl = np.random.uniform(low=-10000, high=10000, size=(i1,))
print(sampl)
reconstimg1 = np.matrix(U[:, :i1]) * np.diag(sigma[:i1]+sampl) * np.matrix(V[:i1, :])
title = "n = %s" % i1
plt.imshow(reconstimg1, cmap='gray')
plt.title(title)
plt.show()
U1, sigma1, V1 = np.linalg.svd(reconstimg1)

reconstimg2 = np.matrix(U[:, :i2]) * np.diag(sigma[:i2]) * np.matrix(V[:i2, :])
title = "n = %s" % i2
plt.imshow(reconstimg2, cmap='gray')
plt.title(title)
plt.show()
U2, sigma2, V2 = np.linalg.svd(reconstimg2)

distcos=1-np.dot(sigma1[:i1],sigma2[:i1])/sqrt(np.dot(sigma1[:i1],sigma1[:i1])*np.dot(sigma2[:i1],sigma2[:i1]))
print(distcos)

#%%

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
#
img1 = Image.open('test.jpeg')
img2 = Image.open('test.jpeg')

imggray1 = img1.convert('LA')
imggray2 = img2.convert('LA')
plt.figure(figsize=(9, 6))
plt.imshow(imggray1);
plt.figure(figsize=(9, 6))
plt.imshow(imggray2);
plt.show()

imgmat1 = np.array(list(imggray1.getdata(band=0)), float)
imgmat1.shape = (imggray1.size[1], imggray1.size[0])
imgmat1 = np.matrix(imgmat1)
plt.figure(figsize=(9,6))
plt.imshow(imgmat1, cmap='gray');
U1, sigma1, V1 = np.linalg.svd(imgmat1)

imgmat2 = np.array(list(imggray2.getdata(band=0)), float)
imgmat2.shape = (imggray2.size[1], imggray2.size[0])
imgmat2 = np.matrix(imgmat2)

imgmat2 += np.random.randn(np.shape(imgmat2)[0],np.shape(imgmat2)[1])*100

plt.imshow(imgmat2, cmap='gray')

plt.figure(figsize=(9,6))
plt.imshow(imgmat2, cmap='gray');
U2, sigma2, V2 = np.linalg.svd(imgmat2)

isvd=50

reconstimg2 = np.matrix(U2[:, :isvd]) * np.diag(sigma2[:isvd]) * np.matrix(V2[:isvd, :])
title = "n = %s" % i2
plt.imshow(reconstimg2, cmap='gray')


distcos=1-np.dot(sigma1[:isvd],sigma2[:isvd])/sqrt(np.dot(sigma1[:isvd],sigma1[:isvd])*np.dot(sigma2[:isvd],sigma2[:isvd]))
print(distcos)



# %%
