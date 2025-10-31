import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.interpolate import griddata

# Open video capture (0 for webcam, or provide video file path)
cap = cv2.VideoCapture(5)

# Get FPS to calculate velocity in pixels/second
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # Default if FPS cannot be determined

# Reduce feature detection for slower, smoother tracking
feature_params = dict(maxCorners=50, qualityLevel=0.3, minDistance=20, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect initial corner points
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

def speed_to_color(speed, min_speed, max_speed):
    """Convert speed to color using a blue(far) to red(close) colormap"""
    if max_speed == min_speed:
        normalized = 0.5
    else:
        normalized = np.clip((speed - min_speed) / (max_speed - min_speed), 0, 1)
    
    # Use OpenCV's COLORMAP_JET for smooth gradient
    color_value = int(normalized * 255)
    color_array = np.uint8([[[color_value]]])
    color_bgr = cv2.applyColorMap(color_array, cv2.COLORMAP_JET)[0][0]
    
    return tuple(map(int, color_bgr))

def create_depth_overlay(frame, positions, speeds, min_speed, max_speed):
    """Create a smooth morphed color overlay based on depth"""
    h, w = frame.shape[:2]
    
    if len(positions) < 3:
        return frame
    
    # Add corner and edge points to ensure full coverage
    border_positions = [
        [0, 0], [w//2, 0], [w-1, 0],
        [0, h//2], [w-1, h//2],
        [0, h-1], [w//2, h-1], [w-1, h-1],
        [w//4, h//4], [3*w//4, h//4],
        [w//4, 3*h//4], [3*w//4, 3*h//4]
    ]
    
    # Use mean speed for border points
    mean_speed = np.mean(speeds)
    border_speeds = [mean_speed] * len(border_positions)
    
    # Combine actual positions with border points
    all_positions = positions + border_positions
    all_speeds = list(speeds) + border_speeds
    
    # Create finer grid for smoother interpolation
    grid_x, grid_y = np.meshgrid(np.linspace(0, w-1, w//2), np.linspace(0, h-1, h//2))
    
    # Prepare points and values for interpolation
    points = np.array(all_positions)
    values = np.array(all_speeds)
    
    # Interpolate speeds across the entire image
    try:
        # Use linear interpolation for smoother results
        grid_speeds = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=mean_speed)
        
        # Fill any remaining NaN values
        mask = np.isnan(grid_speeds)
        if np.any(mask):
            grid_speeds[mask] = mean_speed
        
        # Resize to full resolution with smooth interpolation
        grid_speeds_full = cv2.resize(grid_speeds, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Create color overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        for y in range(h):
            for x in range(w):
                speed = grid_speeds_full[y, x]
                color = speed_to_color(speed, min_speed, max_speed)
                overlay[y, x] = color
        
        # Apply stronger Gaussian blur for very smooth appearance
        overlay = cv2.GaussianBlur(overlay, (31, 31), 0)
        
        # Blend overlay with original frame (more overlay visibility)
        result = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        return result
    except:
        return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    
    # Calculate velocity vectors and speeds
    speeds = []
    positions_new = []
    
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # Displacement in pixels
        dx = new[0] - old[0]
        dy = new[1] - old[1]
        
        # Velocity in pixels per second
        vx = dx * fps
        vy = dy * fps
        
        # Speed magnitude
        speed = np.sqrt(vx**2 + vy**2)
        
        speeds.append(speed)
        positions_new.append([new[0], new[1]])
    
    if len(speeds) >= 3:
        # Apply K-means clustering based on speed
        speeds_array = np.array(speeds).reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(speeds_array)
        
        # Get cluster centers (mean speeds) and sort them
        speed_means = sorted(kmeans.cluster_centers_.flatten())
        
        # Calculate distances between cluster means
        dist_1_2 = speed_means[1] - speed_means[0]
        dist_2_3 = speed_means[2] - speed_means[1]
        
        # Define color scale range
        min_speed = max(0, speed_means[0] - 10)  # Add some padding
        max_speed = speed_means[2] + 10
        
        # Create smooth depth overlay
        depth_map = create_depth_overlay(frame, positions_new, speeds, min_speed, max_speed)
        
        # Draw feature points on top (optional - can comment out for cleaner look)
        for i, pos in enumerate(positions_new):
            speed = speeds[i]
            color = speed_to_color(speed, min_speed, max_speed)
            x, y = int(pos[0]), int(pos[1])
            
            # Draw small circles
            cv2.circle(depth_map, (x, y), 4, color, -1)
            cv2.circle(depth_map, (x, y), 4, (255, 255, 255), 1)
        
        # Create color scale legend
        legend_width = 200
        legend_height = 30
        legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
        
        for x in range(legend_width):
            norm_val = x / legend_width
            speed_val = min_speed + norm_val * (max_speed - min_speed)
            color = speed_to_color(speed_val, min_speed, max_speed)
            legend[:, x] = color
        
        # Add legend to frame
        legend_y = 50
        legend_x = depth_map.shape[1] - legend_width - 20
        
        # Add semi-transparent background for legend
        overlay_bg = depth_map.copy()
        cv2.rectangle(overlay_bg, (legend_x-50, legend_y-10), 
                     (legend_x+legend_width+80, legend_y+legend_height+10), (0, 0, 0), -1)
        depth_map = cv2.addWeighted(depth_map, 0.7, overlay_bg, 0.3, 0)
        
        depth_map[legend_y:legend_y+legend_height, legend_x:legend_x+legend_width] = legend
        
        # Add text labels
        cv2.putText(depth_map, 'FAR', (legend_x - 40, legend_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(depth_map, 'CLOSE', (legend_x + legend_width + 5, legend_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display information with background
        info_box = depth_map.copy()
        cv2.rectangle(info_box, (5, 5), (350, 240), (0, 0, 0), -1)
        depth_map = cv2.addWeighted(depth_map, 0.7, info_box, 0.3, 0)
        
        cv2.putText(depth_map, f'Tracked Points: {len(speeds)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(depth_map, 'Depth Planes (Speed Means):', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display cluster means
        for i in range(3):
            y_pos = 90 + i * 30
            plane_name = ['Far', 'Mid', 'Near'][i]
            plane_color = speed_to_color(speed_means[i], min_speed, max_speed)
            cv2.putText(depth_map, f'{plane_name}: {speed_means[i]:.1f} px/s', 
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, plane_color, 2)
        
        # Display distances
        cv2.putText(depth_map, f'Distance Far-Mid: {dist_1_2:.1f} px/s', 
                    (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        cv2.putText(depth_map, f'Distance Mid-Near: {dist_2_3:.1f} px/s', 
                    (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        
        cv2.imshow('Depth Map (Morphed Color Scale)', depth_map)
    else:
        cv2.imshow('Depth Map (Morphed Color Scale)', frame)
    
    # Update previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    
    # Press 'q' to exit, 'r' to reset tracking
    k = cv2.waitKey(30) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('r'):
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

cap.release()
cv2.destroyAllWindows()
