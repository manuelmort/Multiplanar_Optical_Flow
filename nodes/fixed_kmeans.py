#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import numpy as np
import random
import json


class KMeansNode(Node):
    def __init__(self):
        super().__init__('fixed_kmeans_node')
        
        # Parameters
        self.declare_parameter('centroids', [5, 15, 40])
        self.declare_parameter('buffer_size', 500)
        self.declare_parameter('cluster_frequency', 5.0)  # Hz
        
        self.centroids = self.get_parameter('centroids').value
        self.buffer_size = self.get_parameter('buffer_size').value
        cluster_freq = self.get_parameter('cluster_frequency').value
        
        # Data buffer
        self.vector_buffer = []
        
        # Subscriber
        self.magnitude_sub = self.create_subscription(
            Float32MultiArray,
            'velocity_magnitudes',
            self.magnitude_callback,
            10
        )
        
        # Publishers
        self.cluster_pub = self.create_publisher(String, 'velocity_clusters', 10)
        self.cluster_info_pub = self.create_publisher(Float32MultiArray, 'cluster_centroids', 10)
        
        # Timer for periodic clustering
        self.cluster_timer = self.create_timer(1.0 / cluster_freq, self.cluster_callback)
        
        self.get_logger().info(f'Fixed K-means Node started (centroids={self.centroids}, buffer={self.buffer_size})')    

    def magnitude_callback(self, msg):
        """Receive velocity magnitudes and buffer them"""
        self.vector_buffer.extend(msg.data)
        
        # Keep buffer at max size
        if len(self.vector_buffer) > self.buffer_size:
            self.vector_buffer = self.vector_buffer[-self.buffer_size:]
        
        self.get_logger().debug(f'Buffer size: {len(self.vector_buffer)}')
    
    def cluster_callback(self):
        """Periodically run k-means on buffered data"""
        
        try:
            # Run k-means
            clusters = self.fixed_kmeans(self.vector_buffer,self.centroids)
            
            # Calculate cluster info
            cluster_info = []
            for i, cluster in enumerate(clusters):
                if len(cluster) > 0:
                    centroid = np.mean(cluster)
                    cluster_info.append({
                        'cluster_id': i,
                        'centroid': float(centroid),
                        'size': len(cluster),
                        'min': float(np.min(cluster)),
                        'max': float(np.max(cluster))
                    })
            
            # Publish clusters as JSON string
            cluster_msg = String()
            cluster_msg.data = json.dumps(cluster_info, indent=2)
            self.cluster_pub.publish(cluster_msg)
            
            # Publish centroids as array
            centroids = [info['centroid'] for info in cluster_info]
            centroid_msg = Float32MultiArray()
            centroid_msg.data = centroids
            self.cluster_info_pub.publish(centroid_msg)
            
            # Log results
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
    
    def fixed_kmeans(self,vectors, centroids):
       
        vectors = np.array(vectors)
        centroids = np.array(centroids)
        k = len(centroids) 
        
        clusters = [[] for i in range(k)]

        for i in range(len(vectors)):
            norm = np.linalg.norm(vectors[i])
            closest_idx = np.argmin(np.abs(norm-centroids))

            clusters[closest_idx].append(vectors[i])
        
 
        return clusters


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
