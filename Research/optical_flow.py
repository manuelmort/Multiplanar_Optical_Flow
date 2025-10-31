
#!/usr/bin/env python3
import cv2
import numpy as np

# -------- Camera Setup --------
CAMERA_PATH = "/dev/video4"
cap = cv2.VideoCapture(CAMERA_PATH)
if not cap.isOpened():
    print(f"[ERROR] Cannot open camera at {CAMERA_PATH}")
    exit()

# -------- Parameters --------
feature_params = dict(maxCorners=100,
                      qualityLevel=0.1,
                      minDistance=4,
                      blockSize=5)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# -------- Initialization --------
ret, old_frame = cap.read()
if not ret:
    print("[ERROR] Cannot read initial frame.")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Setup window
cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Optical Flow", 1280, 720)
cv2.waitKey(100)
print("[INFO] Optical flow started — press ESC to quit.")

# -------- Main Loop --------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Optical Flow Calculation ---
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

    if p1 is None or len(p1[st == 1]) < 50:
        p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
        old_gray = gray.copy()
        continue

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    display = frame.copy()
    magnitudes = []
    # --- Draw each motion vector and its distance ---
    for (new, old) in zip(good_new, good_old):
        x1, y1 = new.ravel()
        x0, y0 = old.ravel()

        # Compute displacement vector
        dx, dy = x1 - x0, y1 - y0
        dist = np.sqrt(dx**2 + dy**2)
        magnitudes.append(dist)
        

        # Draw old (blue) and new (red) points
        cv2.circle(display, (int(x0), int(y0)), 2, (255, 0, 0), -1)  # old
        cv2.circle(display, (int(x1), int(y1)), 2, (0, 0, 255), -1)  # new
        cv2.arrowedLine(display, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1, tipLength=0.1)

       # --- Display total average displacement ---
    avg_dist = np.mean(np.linalg.norm(good_new - good_old, axis=1))
    magnitudes = np.array(magnitudes) 
    print(magnitudes)
    cv2.imshow("Optical Flow", display)

    # Update for next iteration
    p0 = good_new.reshape(-1, 1, 2)
    old_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
