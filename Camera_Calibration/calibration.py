import cv2
import numpy as np

# 1. Define checkerboard dimensions
CHECKERBOARD = (9, 6) # Inner corners of a 10x7 checkerboard
SQUARE_SIZE = 24.0    # Exact physical size of one square in mm, 2.4 cm

# 2. Generate 3D object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# 3. Load calibration video
video_path = "/mnt/c/Users/dinay/ComputerVisionAssignment/Camera_Calibration/Calibration_Vid.mp4"
cap = cv2.VideoCapture(video_path)

# Parameters for video processing
frame_count = 0
frame_interval = 30 # Sample every 30th frame to avoid redundant data
gray_shape = None

print("Extracting frames from video...")

cv2.namedWindow('Calibration Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Calibration Frame', 540, 960)

while cap.isOpened():
    ret_val, frame = cap.read()
    if not ret_val:
        break # Video has ended
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    
    # Process only a subset of frames to ensure diverse angles and faster processing
    if frame_count % frame_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]
        
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret:
            objpoints.append(objp)
            
            # Refine corner locations to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Optional visual feedback
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
            cv2.imshow('Calibration Frame', frame)
            cv2.waitKey(1)
            
    frame_count += 1

cap.release()
cv2.destroyAllWindows()

if not objpoints:
    print("No valid checkerboard frames were found in the video.")
else:
    print(f"Successfully extracted {len(objpoints)} frames. Calculating calibration matrix...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

    # 4. Save the output to a YAML file
    save_path = "/mnt/c/Users/dinay/ComputerVisionAssignment/Camera_Calibration/calibration.yaml"
    cv_file = cv2.FileStorage(save_path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("Camera_Matrix", mtx)
    cv_file.write("Distortion_Coefficients", dist)
    cv_file.release()
    print(f"Calibration successful and saved to {save_path}.")