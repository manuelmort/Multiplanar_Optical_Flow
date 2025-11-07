import numpy as np

def ordered_kmeans_descending(vectors, initial_centroids):
    """K-means with descending order constraint: c1 > c2 > c3"""
    vectors = np.array(vectors)
    centroids = np.array(initial_centroids, dtype=float)
    k = len(centroids)

    for _ in range(10):  # iterate a few times until centroids stabilize
        # Assign vectors to clusters
        clusters = [[] for _ in range(k)]
        for vector in vectors:
            norm = np.linalg.norm(vector)
            idx = np.argmin(np.abs(norm - centroids))
            clusters[idx].append(vector)

        # Update centroids
        new_centroids = np.array([
            update_mean(cluster) for cluster in clusters
        ])

        # Enforce descending order
        centroids = np.sort(new_centroids)[::-1]

    return clusters, centroids

def update_mean(cluster):
    if len(cluster) == 0:
        return 0
    return np.mean([np.linalg.norm(v) for v in cluster])

if __name__ == "__main__":
    vectors = [(3,6), (20,3), (1,2), (3,4), (5,20), (20,20), (10,10), (50,50)]
    initial_means = [40, 15, 5]  # High, Medium, Low

    result, final_centroids = ordered_kmeans_descending(vectors, initial_means)

    print(f"Initial means: {initial_means}")
    print(f"Final means: {final_centroids} (still descending!)\n")

    for i, cluster in enumerate(result):
        label = ["High", "Medium", "Low"][i]
        print(f"Cluster {i+1} ({label} - centroid={final_centroids[i]:.2f}):")
        print(f"  Vectors: {cluster}")
        print(f"  Size: {len(cluster)} points\n")

