import cv2 as cv
import numpy as np
import imageio
from scipy.linalg import null_space
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

#1 codes
#Loading the original images
cam1 = cv.imread('epipairs/library1.jpg')
cam2 = cv.imread('epipairs/library2.jpg')

#loading the matches array
matches = np.loadtxt('epipairs/library_matches.txt')

#I assumed here that the first two columns correspond to points in first image
#last columns are for coordinates in second image
#Slicing of matches matrix
cam1_pts = matches[:,:2]
cam2_pts = matches[:,2:]

#2 codes
def estimateFundamental(pts1,pts2):

    n = pts1.shape[1]
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [pts1[0,i]*pts2[0,i],pts1[0,i]*pts2[1,i],pts1[0,i]*pts2[2,i],
        pts1[1,i]*pts2[0,i],pts1[1,i]*pts2[1,i],pts1[1,i]*pts2[2,i],
        pts1[2,i]*pts2[0,i],pts1[2,i]*pts2[1,i],pts1[2,i]*pts2[2,i]]
    
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)

    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U,np.dot(np.diag(S),V))
    F = F/F[2,2]

    return F

def estimateFundamentalNormalized(p1,p2):

    n = p1.shape[1]

    # normalize image coordinates
    # follows the formulation in
    # https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html
    x1 = p1/p1[2]
    # computes the means
    m1 = np.mean(x1[:2],axis=1)
    # computes the scale terms
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*m1[0]],[0,S1,-S1*m1[1]],[0,0,1]])
    x1 = np.dot(T1,x1)
    
    x2 = p2 / p2[2]
    m2 = np.mean(x2[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*m2[0]],[0,S2,-S2*m2[1]],[0,0,1]])
    x2 = np.dot(T2,x2)

    # compute F with the normalized coordinates
    F = estimateFundamental(x1,x2)

    # reverse normalization
    F = np.dot(T1.T,np.dot(F,T2))

    return F/F[2,2]

#This function computes for the distance of a point to a line
def residual(F, p1, p2):

    L2 = np.matmul(F, p1.T).transpose() #epipolar line for camera1
    L2_norm = np.sqrt(L2[:,0]**2 + L2[:,1]**2) #euclidean distance
    L2 = L2 / L2_norm[:,np.newaxis]
    dist = np.multiply(L2, p2).sum(axis = 1) #distance of a point to a line
    return np.mean(np.square(dist))

#Finding the fundamental matrix from both points
#LMEDS is the least median algorithm for normalizing the fundamental matrix
F, mask = cv.findFundamentalMat(cam1_pts,cam2_pts,cv.FM_8POINT)

#Counting the number of inliers
#Inliers are valued 1 in the list
in_count = mask.ravel().tolist().count(1)

#Storing points that are inliers
cam1_pts = np.int32(cam1_pts[mask.ravel()==1])
cam2_pts = np.int32(cam2_pts[mask.ravel()==1])

#Z-column for the point vector
z_col = np.ones((in_count,1))

#Appending the z_col to the x,y point coordinates
pts1 = np.hstack((cam1_pts,z_col))
pts2 = np.hstack((cam2_pts,z_col))

error2 = residual(F,pts1,pts2)
error1 = residual(F.T,pts2,pts1)
print('Residual in Camera 1 (OpenCV): ',error1)
print('Residual in Camera 2 (OpenCV): ',error2)

F = estimateFundamentalNormalized(pts1.T,pts2.T)

error2 = residual(F,pts1,pts2)
error1 = residual(F.T,pts2,pts1)
print('Residual in Camera 1 (Scratch): ',error1)
print('Residual in Camera 2 (Scratch): ',error2)

#3 codes
#This portion is for the putative correspondences using PA2 code
query_img = imageio.imread('epipairs/library1.jpg')
train_img = imageio.imread('epipairs/library2.jpg')

row, col,_ = query_img.shape

#color of corners
red = [255,0,0]

#get points in query image
g_query_img = cv.cvtColor(query_img, cv.COLOR_RGB2GRAY)
query_dst = cv.cornerHarris(g_query_img, 7, 9, 0.05)
#query_dst = cv.dilate(query_dst,None)
query_img[query_dst>0.1*query_dst.max()] = red
query_X, query_Y = np.where(np.all(query_img==red,axis=2))
query_pts = np.column_stack((query_X,query_Y))
query_pts = np.float32(query_pts)

#get points in train image
g_train_img = cv.cvtColor(train_img, cv.COLOR_RGB2GRAY)
train_dst = cv.cornerHarris(g_train_img, 7, 9, 0.05)
#train_dst = cv.dilate(train_dst,None)
train_img[train_dst>0.1*train_dst.max()] = red
train_X, train_Y = np.where(np.all(train_img==red,axis=2))
train_pts = np.column_stack((train_X,train_Y))
train_pts = np.float32(train_pts)

#keypoint conversion
kpsTrain = []
kpsQuery = []

def keyConvert(points):
    arr = []

    for i in points:
        x, y = i
        x, y = float(x), float(y)

        kp = cv.KeyPoint(y,x,10)
        arr.append(kp)
    
    return arr

kpsTrain = keyConvert(train_pts)
kpsQuery = keyConvert(query_pts)

#patch size
size = 8
half = int(size/2)

row0,_ = query_pts.shape  
row1,_ = train_pts.shape

#descriptor arrays
query_ft = []
train_ft = []

def getFeatures(gray_img, points):

    arr = []

    for i in points:
        x, y = i
        x, y = int(x), int(y)

        #Checking if outside image range
        if x-half < 0:
            beginX = 0
            endX = beginX + size
        elif x+half > row:
            endX = row
            beginX = endX - size
        else:
            beginX = x-half
            endX = x+half

        if y-half < 0:
            beginY = 0
            endY = beginY + size
        elif y+half > col:
            endY = col
            beginY = endY - size
        else:
            beginY = y-half
            endY = y+half

        patch = gray_img[beginX:endX,beginY:endY]
        patch = patch.reshape(-1)
        arr = np.append(arr,patch)

    return arr

query_ft = getFeatures(g_query_img, query_pts)
train_ft = getFeatures(g_train_img, train_pts)

#reshaping the vectorized patches
query_ft = query_ft.reshape(row0,-1)
train_ft = train_ft.reshape(row1,-1)

query_ft = np.float32(query_ft)
train_ft = np.float32(train_ft)

#computing the Euclidean distance
bf = cv.BFMatcher(cv.NORM_L2,crossCheck=True)
matches = bf.match(train_ft,query_ft)
#matches = sorted(matches, key = lambda x:x.distance)
#matches = matches[:100]

kpsT = np.float32([kp.pt for kp in kpsTrain])
kpsQ = np.float32([kp.pt for kp in kpsQuery])

#convert points to integer
ptsA = np.int32([kpsT[m.queryIdx] for m in matches])
ptsB = np.int32([kpsQ[m.trainIdx] for m in matches])

new_F, new_mask = cv.findFundamentalMat(ptsA,ptsB,cv.RANSAC,-100)

ptsA = ptsA[new_mask.ravel()==1]
ptsB = ptsB[new_mask.ravel()==1]

new_pts1 = np.hstack((ptsA,np.ones((ptsA.shape[0],1))))
new_pts2 = np.hstack((ptsB,np.ones((ptsB.shape[0],1))))

new_error2 = residual(new_F,new_pts1,new_pts2)
new_error1 = residual(new_F.T,new_pts2,new_pts1)
print('Residual in Camera 1 (Harris-OpenCV): ',new_error1)
print('Residual in Camera 2 (Harris-OpenCV): ',new_error2)

new_F = estimateFundamentalNormalized(new_pts1.T,new_pts2.T)

new_error2 = residual(new_F,new_pts1,new_pts2)
new_error1 = residual(new_F.T,new_pts2,new_pts1)
print('Residual in Camera 1 (Harris-Scratch): ',new_error1)
print('Residual in Camera 2 (Harris-Scratch): ',new_error2)

#4 codes
cam_mat1 = np.loadtxt('epipairs/library1_camera.txt')
cam_mat2 = np.loadtxt('epipairs/library2_camera.txt')

#computes for the null space of the camera projection matrix
#null space of the matrix forms the center of the camera
center1 = null_space(cam_mat1)
center2 = null_space(cam_mat2)

def leastsquareTriangulation(p1,mat1,p2,mat2):
    
    #required matrix for solving the homogenous equation AX=b 
    A = np.zeros((4,3))
    b = np.zeros((4,1))

    #Matrices for efficient multiplication
    dummy1 = np.array(-np.eye(2,3))
    dummy2 = np.array(-np.eye(2,3))

    tri_arr = []

    for i in range(len(p1)):
        dummy1[:,2] = p1[i,:2]
        dummy2[:,2] = p2[i,:2]

        #constructing values of the A matrix
        A[:2,:] = dummy1.dot(mat1[:3,:3])
        A[2:,:] = dummy2.dot(mat2[:3,:3])

        b[:2,:] = dummy1.dot(mat1[:3,3:])
        b[2:,:] = dummy2.dot(mat2[:3,3:])
        
        #Least square for solving the homogenous system
        X = np.linalg.lstsq(A,-b,rcond=None)[0]
        tri_arr = np.append(tri_arr,X)

    return tri_arr

def evaluate3DPoints(pts1,camera_matrix,points3d):
    #computing for the residuals of the image points and estimated world points
    world = camera_matrix.dot(points3d.T)
    world = world/world[2]
    res = np.mean(np.square(np.linalg.norm(pts1 - world.T)))

    return res

#3D points obtained from triangulation of ground truth data
points_3d = leastsquareTriangulation(pts1,cam_mat1,pts2,cam_mat2)
points_3d  = points_3d.reshape(-1,3)

#3D points converted to 4D
coordinate_3d = np.hstack((points_3d,np.ones((len(points_3d),1))))

res1_3d = evaluate3DPoints(pts1,cam_mat1,coordinate_3d)
res2_3d = evaluate3DPoints(pts2,cam_mat2,coordinate_3d)
print('3D Residual for Camera 1: ',res1_3d)
print('3D Residual for Camera 2: ',res2_3d)

#For plotting. Getting the x, y, z values
x = points_3d[:,0]
y = points_3d[:,1]
z = points_3d[:,2]

#For plotting and visualizing
camera_centers = np.vstack((center1.T, center2.T))
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5,label='Points')
ax.scatter(camera_centers[:, 0], camera_centers[:, 1], 
           camera_centers[:, 2], c='g', s=90, 
           marker='^', label='Camera Centers')
ax.legend(loc='best')
plt.show()

#5 codes
#3D points obtained from triangulation of putative matches
new_points_3d = leastsquareTriangulation(new_pts1,cam_mat1,new_pts2,cam_mat2)
new_points_3d = new_points_3d.reshape(-1,3)

#Conversion to 4D points
new_coordinate_3d = np.hstack((new_points_3d,np.ones((len(new_points_3d),1))))

new_res1_3d = evaluate3DPoints(new_pts1,cam_mat1,new_coordinate_3d)
new_res2_3d = evaluate3DPoints(new_pts2,cam_mat2,new_coordinate_3d)
print('Harris 3D Residual for Camera 1: ',new_res1_3d)
print('Harris 3D Residual for Camera 2: ',new_res2_3d)

#For plotting 
x1 = new_points_3d[:,0]
y1 = new_points_3d[:,1]
z1 = new_points_3d[:,2]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x1, y1, z1, c=z1, cmap='viridis', linewidth=0.5,label='Points')
ax.scatter(camera_centers[:, 0], camera_centers[:, 1], 
           camera_centers[:, 2], c='g', s=90, 
           marker='^', label='Camera Centers')
ax.legend(loc='best')
plt.show()