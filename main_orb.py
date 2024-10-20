import cv2
import time
import os
import glob
import numpy as np
import open3d as o3d
import scipy
import scipy.optimize
import concurrent.futures

cap = cv2.VideoCapture("./KITTI dataset sequence 07 video.mp4")

# Create ORB detector
orb = cv2.ORB_create(nfeatures=20)
# Brute Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

vis = o3d.visualization.Visualizer()
vis.create_window(height=480, width=640)

# initialize pointcloud instance.
pcd = o3d.geometry.PointCloud()

# Add the point cloud to the visualization window
vis.add_geometry(pcd)

# Camera intrinsic parameters
K = np.array([[9.842439e+02, 0.000000e+00, 6.900000e+02],
              [0.000000e+00, 9.808141e+02, 2.331966e+02],
              [0.000000e+00, 0.000000e+00, 1.000000e+00]])

D = np.array([-3.728755e-01, 2.037299e-01, 2.219027e-03, 1.383707e-03, -7.233722e-02])

R_world = np.eye(3)  # Identity rotation
T_world = np.zeros(3)  # Zero translation

map_started = False
Tracking = False

def BA(R, t, points_2d, points_3d):
  def project(R, t, X):
    points, _ =  cv2.projectPoints(points_3d, R, t, K, np.zeros(4))

    return points
  
  def reproj_error(params):
    R = params[:9].reshape(3,3)
    t = np.array([params[9:12]]).T
    pts3d = params[12:].reshape(3,-1)
    reproj_error = points_2d - project(R,t,pts3d)[:,0,:]
    return reproj_error
  
  def huber_loss(errors, delta):
    abs_errors = np.abs(errors)
    loss = np.where(abs_errors <= delta,
                    0.5 * errors**2,
                    delta * (abs_errors - 0.5 * delta))
    
    return loss

  def cost(params):
    sigma = 1.0
    cov_matrices = sigma**2 * np.eye(2)
    
    errors = reproj_error(params)
    weighted_errors = []
    for j in range(errors.shape[0]):
      cov_inv = np.linalg.inv(cov_matrices)
      weighted_error = cov_inv @ errors.T[:,j]
      weighted_errors.append(huber_loss(weighted_error, 1))

    return np.concatenate(weighted_errors)
      
  rvec = R.flatten()
  initial_params = np.hstack((rvec,t.T[0],points_3d.flatten()), dtype=np.float64)

  result = scipy.optimize.least_squares(cost, initial_params, method = 'trf')
  
  if result:
    res = result.x
    R = res[:9].reshape(3,3)
    t = res[9:12].T
    pts3d = res[12:].reshape(3,-1)

    return R,t,pts3d
  
  return None
  
def map_init(frame_ref, frame_curr):
  global R_world, T_world
  def compute_homography(pts_r, pts_c):
    H, inliers = cv2.findHomography(pts_r, pts_c, cv2.RANSAC)
    return H, inliers
  
  def compute_fundamental(pts_r, pts_c):
    F, inliers = cv2.findFundamentalMat(pts_r, pts_c, cv2.FM_RANSAC)
    return F, inliers

  def recover_motion_from_homography(H, pts1, pts2):
    _, Rs, Ts, normals = cv2.decomposeHomographyMat(H, K)

    best_solution = None
    max_points_in_front = 0

    for i in range(len(Rs)):
      R = Rs[i]
      T = Ts[i]

      # Triangulation using R and T
      P1 = np.hstack((np.eye(3), np.zeros((3,1))))
      P2 = np.hstack((R, T.reshape(-1,1)))
      P2 = K @ P2

      pts1_homog = cv2.convertPointsToHomogeneous(pts1).reshape(-1,3)
      pts2_homog = cv2.convertPointsToHomogeneous(pts2).reshape(-1,3)

      points_4d = cv2.triangulatePoints(K @ P1, P2, pts1_homog.T[:2],pts2_homog.T[:2])
      points_3d = points_4d[:3]/points_4d[3]

      points_in_front = np.sum(points_3d[2, :] > 0)
      if points_in_front > max_points_in_front:
        max_points_in_front = points_in_front
        best_solution = (R,T, points_3d)

    return best_solution
  
  def recover_motion_from_fundamental(F, pts1, pts2):
    E = K.T @ F @ K

    _, R, T, mask = cv2.recoverPose(E, pts1, pts2, K)

    # Define projection matrices for the two camera views
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for camera 1
    P2 = K @ np.hstack((R, T.reshape(-1, 1))) # Projection matrix for camera 2

    # Step 4: Triangulate points
    points4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)  # Shape (4, N)
    points3D = points4D[:3] / points4D[3]  # Convert to non-homogeneous coordinates

    return R, T, points3D

  # find correspondences
  kp_c, dp_c, img_curr = frame_curr
  kp_r, dp_r, img_ref = frame_ref

  matches = bf.match(dp_r, dp_c)
  matches = sorted(matches, key=lambda x: x.distance)
  
  if len(matches) < 8:
    return False

  # Get matched points
  pts_r = np.float32([kp_r[m.queryIdx].pt for m in matches])
  pts_c = np.float32([kp_c[m.trainIdx].pt for m in matches])
  
  with concurrent.futures.ThreadPoolExecutor() as executor:
    # Schedule both functions to run in parallel
    future_homography = executor.submit(compute_homography, pts_r, pts_c)
    future_fundamental = executor.submit(compute_fundamental, pts_r, pts_c)

    # Wait for both threads to complete and retrieve the results
    H, inliers_h = future_homography.result()  # Homography result
    F, inliers_f = future_fundamental.result()  # Fundamental matrix result
  
  SH = np.sum(inliers_h)
  SF = np.sum(inliers_f)

  homography = False
  if SH/(SH+SF) > 0.45:
    mat = H
    inliers = inliers_h
    homography = True
  else:
    mat = F
    inliers = inliers_f

  try:
    if homography:
      R,T, pts3d = recover_motion_from_homography(mat, pts_r, pts_c)
    else:
      R,T, pts3d = recover_motion_from_fundamental(mat, pts_r, pts_c)
  except TypeError:
    return False
  
  R,t,pts3d = BA(R,T, pts_c, pts3d)

def tracking(frame):
  global prev_frame, map_started, Tracking

  def estimatePose(frame_ref, frame):
    pass

  def relocalize(frame):
    pass

  kp, dp = orb.detectAndCompute(frame, None)
  frame = kp, dp, frame

  if not map_started:
    if prev_frame is None:
      prev_frame = frame
      return
    else:
      map_init(prev_frame, frame)

  if not Tracking:
    relocalize(frame)
  else:
    estimatePose(frame_ref, frame)
  
  # track local map

  # new keyframe decision

prev_frame = None
while cap.isOpened():
  ret_frame, frame = cap.read()
  if not ret_frame:
      break
  
  cv2.imshow("Live", frame)

  curr_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  kp_curr, dp_curr = orb.detectAndCompute(curr_img, None)
  frame = kp_curr, dp_curr, frame
  if prev_frame is not None:
    
    map_init(prev_frame, frame)
      
  prev_frame = frame
  
  vis.poll_events()
  vis.update_renderer()

  cv2.waitKey(1)
