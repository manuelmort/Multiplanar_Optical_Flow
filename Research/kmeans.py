import numpy as np
import random

def fixed_kmeans(vectors, centroids):
    # 1D kmeans clustering vectors using their magnitudes
    #Converting to numpy array for easier calculations

    vectors = np.array(vectors)
    
    centroids = np.array(centroids) 
    
    k = len(centroids)
    # randoming selecting k points as starting centroids
    
    clusters = [[] for i in range(k)] # Initializing size
    
    for i in range(len(vectors)):

        norm = np.linalg.norm(vectors[i])    
        closest_idx = np.argmin(np.abs(norm-centroids))
       
        clusters[closest_idx].append(vectors[i])
        

    return clusters


if __name__ == "__main__":
    vectors = [(1,2),(3,4), (5,20), (50,50)]

    fixed_means = [5, 15, 40] 
    result = fixed_kmeans(vectors, fixed_means)

    for i, cluster in enumerate(result):
        print(f"Cluster {i+1}: {cluster}")
        print(f"Size:{len(cluster)} points") 

