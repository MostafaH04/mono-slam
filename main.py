import cv2
import time
import os
import glob
import numpy as np

import open3d as o3d

def add_ones(x):
  return np.concatenate([x, np.ones((x.shape[0], 1))], axis = 1)

img_date = "2011_09_26"
img_file = "image_00"

image_dir = os.path.join(".\data", f"{img_date}_imgs\{img_file}")

images = glob.glob(f"{image_dir}\data\*.png")

# Create orb detect
orb = cv2.ORB_create(nfeatures=1000)
# Brute Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# camera intrinsic parameters
K = np.array(
  [[9.842439e+02, 0.000000e+00, 6.900000e+02],
   [0.000000e+00, 9.808141e+02, 2.331966e+02],
   [0.000000e+00, 0.000000e+00, 1.000000e+00]
])

with open(f"{image_dir}\\timestamps.txt", "r") as timestamp_file:
  prev_time = None
  prev_img = None
  prev_kp = None
  prev_dp = None
  pose = np.eye(4)
  mapp_pts = []
  for img_num, time_stamp in enumerate(timestamp_file):
    curr_time = float(time_stamp[17:-1])
    curr_img = cv2.imread(images[img_num])

    kp, dp = orb.detectAndCompute(curr_img, None)

    img = cv2.drawKeypoints(curr_img, kp, None, color = (0,255,0), flags = 0)

    cv2.imshow("Live", img)


    if prev_time is not None:
      dt = curr_time - prev_time

      matches = bf.match(prev_dp, dp)
      matches = sorted(matches, key = lambda x: x.distance)

      # get mactched points
      pts1 = np.float32([kp[m.queryIdx].pt for m in matches])
      pts2 = np.float32([prev_kp[m.trainIdx].pt for m in matches])

      Fund_mat, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

      # essential matrix
      E = K.T @ Fund_mat @ K

      pts1_inliers = np.delete(pts1, np.invert(inliers.T.astype(np.bool_)[0]), 0)
      pts2_inliers = np.delete(pts2, np.invert(inliers.T.astype(np.bool_)[0]), 0)
      _, R, t, mask = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)

      T = np.eye(4)
      T[0:3,0:3] = R
      T[0:3,3] = t.T

      curr_pose = pose @ T

      ret = np.zeros((pts1.shape[0], 4))
      pose1 = np.linalg.inv(pose)
      pose2 = np.linalg.inv(curr_pose)
      
      for i, p in enumerate(zip(add_ones(pts1), 
                              add_ones(pts2))):
        A = np.zeros((4, 4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]

      ret /= ret[:, 3:]
      good_pts4d =   (np.abs(ret[:, 3]) > 0.005) & (ret[:, 2] > 0)

      mapp_pts += [p for i, p in enumerate(ret) if good_pts4d[i]]
      
      cv2.waitKey(1)
    
    prev_kp = kp
    prev_dp = dp
    prev_img = curr_img
    prev_time = curr_time

  mapp_pts = np.array(mapp_pts)

  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(mapp_pts[:,:3])
  colors = np.zeros((mapp_pts.shape[0], 3))
  colors[:, 1] = 1

  pcd.colors = o3d.utility.Vector3dVector(colors)

  o3d.visualization.draw_geometries([pcd])

    