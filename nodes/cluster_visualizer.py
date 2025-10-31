#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import json


class ClusterVisualizerNode(Node):
    def __init__(self):
        super().__init__('cluster_visualizer_node')
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Subscribers
        self.cluster_sub = self.create_subscription(
            String,
            'velocity_clusters',
            self.cluster_callback,
            10
        )
        
        self.image_sub = self.create_subscription(
            Image,
            'optical_flow_image',
            self.image_callback,
            10
        )
        
        self.vector_sub = self.create_subscription(
            Float32MultiArray,
            'velocity_vectors',
            self.vectors_callback,
            10
        )
        
        # Data storage
        self.current_clusters = None
        self.current_frame = None
        self.current_vectors = []
        
        # Create windows
        cv2.namedWindow("Cluster 1 - Low Velocity", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Cluster 2 - Medium Velocity", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Cluster 3 - High Velocity", cv2.WINDOW_NORMAL)
        
        cv2.resizeWindow("Cluster 1 - Low Velocity", 640, 480)
        cv2.resizeWindow("Cluster 2 - Medium Velocity", 640, 480)
        cv2.resizeWindow("Cluster 3 - High Velocity", 640, 480)
        
        # Timer for updating display
        self.timer = self.create_timer(0.1, self.update_display)
        
        self.get_logger().info('Cluster Visualizer Node started')
    
    def vectors_callback(self, msg):
        """Store current magnitudes"""
        self.vectors_magnitudes = list(msg.data)
    
    def cluster_callback(self, msg):
        """Receive cluster information"""
        try:
            self.current_clusters = json.loads(msg.data)
            self.get_logger().debug(f'Received {len(self.current_clusters)} clusters')
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse cluster data: {e}')
    
    def image_callback(self, msg):
        """Receive optical flow image"""
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
    
    def update_display(self):
        """Update the three cluster windows"""
        if self.current_clusters is None or self.current_frame is None:
            return
        
        # Sort clusters by centroid (low to high velocity)
        sorted_clusters = sorted(self.current_clusters, key=lambda x: x['centroid'])
        
        # Create visualization for each cluster
        for i in range(3):
            if i < len(sorted_clusters):
                cluster = sorted_clusters[i]
                window_name = f"Cluster {i+1} - "
                
                if i == 0:
                    window_name += "Low Velocity"
                elif i == 1:
                    window_name += "Medium Velocity"
                else:
                    window_name += "High Velocity"
                
                # Create visualization
                viz = self.create_cluster_visualization(cluster, self.current_frame.copy())
                cv2.imshow(window_name, viz)
            else:
                # Show empty frame if cluster doesn't exist
                window_name = f"Cluster {i+1} - "
                if i == 0:
                    window_name += "Low Velocity"
                elif i == 1:
                    window_name += "Medium Velocity"
                else:
                    window_name += "High Velocity"
                
                empty = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(empty, "No data", (250, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(window_name, empty)
        
        cv2.waitKey(1)
    
    def create_cluster_visualization(self, cluster, frame):
        """Create a visualization for a single cluster"""
        h, w = frame.shape[:2]
        viz = np.zeros((h, w + 300, 3), dtype=np.uint8)  # Frame + info panel
        
        # Copy frame to left side
        viz[:h, :w] = frame
        
        # Info panel on right
        info_panel = viz[:, w:]
        
        # Draw cluster information
        y_offset = 50
        cv2.putText(info_panel, f"Cluster {cluster['cluster_id'] + 1}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 40
        cv2.putText(info_panel, f"Centroid:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 25
        cv2.putText(info_panel, f"{cluster['centroid']:.2f} px/frame", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 40
        cv2.putText(info_panel, f"Size: {cluster['size']} points", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y_offset += 40
        cv2.putText(info_panel, f"Range:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 25
        cv2.putText(info_panel, f"[{cluster['min']:.2f}, {cluster['max']:.2f}]", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
        
        # Draw a simple histogram
        y_offset += 60
        cv2.putText(info_panel, "Distribution:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Simple bar representation
        bar_height = int((cluster['size'] / max(1, sum(c['size'] for c in self.current_clusters))) * 150)
        cv2.rectangle(info_panel, (10, y_offset + 20), (280, y_offset + 20 + bar_height), 
                     (0, 255, 255), -1)
        cv2.rectangle(info_panel, (10, y_offset + 20), (280, y_offset + 170), 
                     (100, 100, 100), 2)
        
        return viz
    
    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ClusterVisualizerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
