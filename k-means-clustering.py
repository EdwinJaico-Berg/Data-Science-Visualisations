import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sklearn.datasets import make_blobs

def get_random_centroids(k: int, x: np.array, y: np.array) -> np.array:
    """Generate random centroids."""
    # Get the min and max x, y values 
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    centroids = []

    for _ in range(k):
        x_c = np.random.choice(np.arange(x_min, x_max, 0.1))
        y_c = np.random.choice(np.arange(y_min, y_max, 0.1))
        centroids.append([x_c, y_c])
        
    return np.array(centroids)

def assign_centroids(X: np.array, centroids: np.array) -> np.array:
    """Assign the cluster label to data point."""
    # Create an array to store the cluster labels
    cluster_labels = np.zeros(X.shape[0])
    
    # For each data point, calculate the distance to each centroid and assign the point to the closest cluster
    for index, x in enumerate(X):
        distances = np.linalg.norm(x - centroids, axis=1)
        cluster_labels[index] = np.argmin(distances)
    
    return cluster_labels

def calculate_centroids(X, k, cluster_labels) -> np.array:
    """Calculate the updated centroid position."""
    # Create an array to store the new centroids
    new_centroids = np.zeros((k, X.shape[1]))

    # For each cluster calculate the mean of the data points - this will be the new centroid
    for i in range(k):
        points = X[np.where(cluster_labels==i)]
        new_centroids[i] = np.mean(points, axis=0)

    return new_centroids

def update(X, k, cluster_labels, cent, scat):
    # Calculate new centroids
    new_centroids = calculate_centroids(X, k, cluster_labels)

    # Update the positions of the centroids
    cent.set_offsets(new_centroids)

    scat.set_array(assign_centroids(X, new_centroids))

    centroids = new_centroids

    return centroids


def main(k: int=3, max_iters: int=1000) -> None:
    X, _ = make_blobs(n_samples=100, centers=k, random_state=170)

    x, y = X[:, 0], X[:, 1]
    
    centroids = get_random_centroids(k, x, y)

    cluster_labels = assign_centroids(X, centroids)
    
    fig, ax = plt.subplots()

    scat = ax.scatter(x=x, y=y, c=cluster_labels)
    cent = ax.scatter(x=centroids[:,0], y=centroids[:,1], c='r')

    anim = animation.FuncAnimation(fig, update(X, k, cluster_labels, cent, scat), frames=range(1000), interval=500)

    plt.show()


if __name__ == "__main__":
    main()