#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class OpticalFlowNode(Node):
    def __init__(self):
        super().__init__('optical_flow_node')
        
        # Publishers
        self.magnitude_pub = self.create_publisher(Float32MultiArray, 'velocity_magnitudes', 10)
        self.image_pub = self.create_publisher(Image, 'optical_flow_image', 10)
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Parameters
        self.declare_parameter('camera_path', '/dev/video5')
        self.declare_parameter('max_corners', 100)
        self.declare_parameter('quality_level', 0.1)
        self.declare_parameter('min_distance', 4)
        
        camera_path = self.get_parameter('camera_path').value
        max_corners = self.get_parameter('max_corners').value
        quality_level = self.get_parameter('quality_level').value
        min_distance = self.get_parameter('min_distance').value
        
        # Camera setup
        self.cap = cv2.VideoCapture(camera_path)
        if not self.cap.isOpened():
            self.get_logger().error(f'Cannot open camera at {camera_path}')
            raise RuntimeError('Camera initialization failed')
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=5
        )
        
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Initialize first frame
        ret, old_frame = self.cap.read()
        if not ret:
            self.get_logger().error('Cannot read initial frame')
            raise RuntimeError('Frame initialization failed')
        
        self.old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
        
        # Timer for processing frames (~30 Hz)
        self.timer = self.create_timer(0.033, self.process_frame)
        
        self.get_logger().info('Optical Flow Node started')
    
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to read frame')
            return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_gray, gray, self.p0, None, **self.lk_params
        )
        
        # Redetect features if too few
        if p1 is None or len(p1[st == 1]) < 50:
            self.p0 = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.old_gray = gray.copy()
            return
        
        good_new = p1[st == 1]
        good_old = self.p0[st == 1]
        
        display = frame.copy()
        magnitudes = []
        
        # Draw motion vectors
        for new, old in zip(good_new, good_old):
            x1, y1 = new.ravel()
            x0, y0 = old.ravel()
            
            # Compute displacement
            dx, dy = x1 - x0, y1 - y0
            dist = np.sqrt(dx**2 + dy**2)
            magnitudes.append(dist)
            
            # Draw visualization
            cv2.circle(display, (int(x0), int(y0)), 2, (255, 0, 0), -1)  # old
            cv2.circle(display, (int(x1), int(y1)), 2, (0, 0, 255), -1)  # new
            cv2.arrowedLine(display, (int(x0), int(y0)), (int(x1), int(y1)), 
                          (0, 255, 0), 1, tipLength=0.1)
        
        # Publish magnitudes
        mag_msg = Float32MultiArray()
        mag_msg.data = magnitudes
        self.magnitude_pub.publish(mag_msg)
        
        # Publish visualization image
        try:
            img_msg = self.bridge.cv2_to_imgmsg(display, encoding='bgr8')
            self.image_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish image: {e}')
        
        # Update for next iteration
        self.p0 = good_new.reshape(-1, 1, 2)
        self.old_gray = gray.copy()
        
        # Log average velocity
        if len(magnitudes) > 0:
            avg_vel = np.mean(magnitudes)
            self.get_logger().info(f'Avg velocity: {avg_vel:.2f} px/frame, Features: {len(magnitudes)}')
    
    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
