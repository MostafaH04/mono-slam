import cv2
import time
import os
import glob
import numpy as np
import open3d as o3d

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def filter_points_by_range(points, min_val=-200, max_val=200):
    # Apply the filter to keep only points where both x and y are within the range
    mask = (points[:, 0] >= min_val) & (points[:, 0] <= max_val) & \
           (points[:, 1] >= min_val) & (points[:, 1] <= max_val)
    return points[mask]

img_date = "2011_09_26"
img_file = "image_00"

image_dir = os.path.join(".\\data", f"{img_date}_imgs\\{img_file}")
images = glob.glob(f"{image_dir}\\data\\*.png")

cap = cv2.VideoCapture("./KITTI dataset sequence 07 video.mp4")

# Create ORB detector
orb = cv2.ORB_create(nfeatures=500)
# Brute Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Camera intrinsic parameters
K = np.array([[9.842439e+02, 0.000000e+00, 6.900000e+02],
              [0.000000e+00, 9.808141e+02, 2.331966e+02],
              [0.000000e+00, 0.000000e+00, 1.000000e+00]])

prev_time = None
prev_img = None
prev_kp = None
prev_dp = None
pose = np.eye(4)
mapp_pts = []

count = 0

# create visualizer and window.
vis = o3d.visualization.Visualizer()
vis.create_window(height=480, width=640)

# initialize pointcloud instance.
pcd = o3d.geometry.PointCloud()
points = np.random.rand(10, 3)
pcd.points = o3d.utility.Vector3dVector(points)

# Add the point cloud to the visualization window
vis.add_geometry(pcd)

while cap.isOpened():
    ret_frame, frame = cap.read()
    if not ret_frame:
        break

    curr_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp, dp = orb.detectAndCompute(curr_img, None)

    output_image = cv2.drawKeypoints(curr_img, kp, 0, (0, 255, 0), 
                                     flags=0)
  
    cv2.imshow("Live", output_image)

    if prev_time is not None:
        matches = bf.match(prev_dp, dp)
        matches = sorted(matches, key=lambda x: x.distance)

        # Get matched points
        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

        Fund_mat, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        # Essential matrix
        E = K.T @ Fund_mat @ K

        pts1_inliers = np.delete(pts1, np.invert(inliers.T.astype(np.bool_)[0]), 0)
        pts2_inliers = np.delete(pts2, np.invert(inliers.T.astype(np.bool_)[0]), 0)
        _, R, t, mask = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)

        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t.T

        curr_pose = pose @ T

        ret = np.zeros((pts1.shape[0], 4))
        pose1 = np.linalg.inv(pose)
        pose2 = np.linalg.inv(curr_pose)

        for i, p in enumerate(zip(add_ones(pts1), add_ones(pts2))):
            A = np.zeros((4, 4))
            A[0] = p[0][0] * pose1[2] - pose1[0]
            A[1] = p[0][1] * pose1[2] - pose1[1]
            A[2] = p[1][0] * pose2[2] - pose2[0]
            A[3] = p[1][1] * pose2[2] - pose2[1]
            _, _, vt = np.linalg.svd(A)
            ret[i] = vt[3]

        ret /= ret[:, 3:]
        good_pts4d = (np.abs(ret[:, 3]) > 0.005) & (ret[:, 2] > 0)
        mapp_pts = [p for i, p in enumerate(ret) if good_pts4d[i]]

        if len(mapp_pts) != 0:
            # Update the point cloud data and refresh the visualization
            mapp_pts_np = np.array(mapp_pts)
            filtered_mapp_pts = filter_points_by_range(mapp_pts_np)
            print(filtered_mapp_pts[:,:3])
            pcd.points.extend(filtered_mapp_pts[:,:3])

            vis.update_geometry(pcd)
        
        vis.poll_events()
        vis.update_renderer()

    prev_kp = kp
    prev_dp = dp
    prev_img = curr_img
    prev_time = 1

    cv2.waitKey(1)

# Cleanup Open3D visualization and OpenCV windows
vis.destroy_window()
cap.release()
cv2.destroyAllWindows()


