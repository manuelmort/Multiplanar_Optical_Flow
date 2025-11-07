#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import numpy as np
import json

class KMeansNode(Node):
    def __init__(self):
        super().__init__('fixed_kmeans_node')
        
        # Parameters
        self.declare_parameter('centroids', [40, 15, 5])  # descending order
        self.declare_parameter('buffer_size', 500)
        self.declare_parameter('cluster_frequency', 5.0)
        
        self.centroids = self.get_parameter('centroids').value
        self.buffer_size = self.get_parameter('buffer_size').value
        cluster_freq = self.get_parameter('cluster_frequency').value
        
        self.vector_buffer = []
        
        # ROS I/O
        self.magnitude_sub = self.create_subscription(
            Float32MultiArray,
            'velocity_magnitudes',
            self.magnitude_callback,
            10
        )
        self.cluster_pub = self.create_publisher(String, 'velocity_clusters', 10)
        self.cluster_info_pub = self.create_publisher(Float32MultiArray, 'cluster_centroids', 10)
        
        # Timer
        self.cluster_timer = self.create_timer(1.0 / cluster_freq, self.cluster_callback)
        self.get_logger().info(f'Fixed K-means Node started (centroids={self.centroids}, buffer={self.buffer_size})')    

    def magnitude_callback(self, msg):
        self.vector_buffer.extend(msg.data)
        if len(self.vector_buffer) > self.buffer_size:
            self.vector_buffer = self.vector_buffer[-self.buffer_size:]
        self.get_logger().debug(f'Buffer size: {len(self.vector_buffer)}')
    
    def cluster_callback(self):
        if len(self.vector_buffer) == 0:
            return
        
        try:
            clusters, centroids = self.fixed_kmeans(self.vector_buffer, self.centroids)
            self.centroids = centroids  # persist for next round
            
            cluster_info = []
            for i, cluster in enumerate(clusters):
                if len(cluster) == 0:
                    continue
                centroid = float(np.mean(cluster))
                cluster_info.append({
                    'cluster_id': i,
                    'centroid': centroid,
                    'size': len(cluster),
                    'min': float(np.min(cluster)),
                    'max': float(np.max(cluster))
                })
            
            # Publish clusters JSON
            cluster_msg = String()
            cluster_msg.data = json.dumps(cluster_info, indent=2)
            self.cluster_pub.publish(cluster_msg)
            
            # Publish centroid array
            centroid_msg = Float32MultiArray()
            centroid_msg.data = [c['centroid'] for c in cluster_info]
            self.cluster_info_pub.publish(centroid_msg)
            
            # Log
            self.get_logger().info(f'Clustered {len(self.vector_buffer)} points into {len(cluster_info)} clusters')
            for info in cluster_info:
                self.get_logger().info(
                    f"Cluster {info['cluster_id']}: "
                    f"centroid={info['centroid']:.2f}, "
                    f"size={info['size']}, "
                    f"range=[{info['min']:.2f}, {info['max']:.2f}]"
                )
        
        except Exception as e:
            self.get_logger().error(f'K-means failed: {e}')
    
    def fixed_kmeans(self, vectors, centroids):
        vectors = np.array(vectors)
        centroids = np.array(centroids, dtype=float)
        k = len(centroids)
        
        for _ in range(5):  # few iterations
            clusters = [[] for _ in range(k)]
            
            # assign step
            for v in vectors:
                idx = np.argmin(np.abs(v - centroids))
                clusters[idx].append(v)
            
            # update step
            new_centroids = np.array([self.update_mean(c) for c in clusters])
            centroids = np.sort(new_centroids)[::-1]  # enforce descending
        
        return clusters, centroids
    
    def update_mean(self, cluster):
        return np.mean(cluster) if len(cluster) > 0 else 0.0

def main(args=None):
    rclpy.init(args=args)
    node = KMeansNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

