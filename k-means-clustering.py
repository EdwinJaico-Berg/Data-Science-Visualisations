import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display, clear_output
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

def calculate_centroids(X: np.array, k: int, cluster_labels: np.array) -> np.array:
    """Calculate the updated centroid position."""
    # Create an array to store the new centroids
    new_centroids = np.zeros((k, X.shape[1]))

    # For each cluster calculate the mean of the data points - this will be the new centroid
    for i in range(k):
        points = X[np.where(cluster_labels==i)]
        new_centroids[i] = np.mean(points, axis=0)

    return new_centroids

def plot(x: np.array, y: np.array, cluster_labels:np.array, centroids:np.array) -> None:
    """Displays the scatter plot."""
    clear_output(wait=True)
    display(plt.gcf())
    plt.clf()
    plt.title("K-Means Clustering")
    sns.scatterplot(x=x, y=y, hue=cluster_labels)
    sns.scatterplot(x=centroids[:,0], y=centroids[:,1])
    plt.legend('',frameon=False)
    plt.xticks([])
    plt.yticks([])
    plt.show(block=False)
    plt.pause(1)


def main() -> None:

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python k-means-clustering.py [clusters] [k] [max_iters]")

    # Parse command line arguments
    clusters = int(sys.argv[1])
    k = int(sys.argv[2])
    max_iters = int(sys.argv[3]) if len(sys.argv) == 4 else 100

    # Create the blobs
    X, _ = make_blobs(n_samples=1000, centers=clusters)

    # Extract x and y values
    x, y = X[:, 0], X[:, 1]
    
    # Generate the initial random centroids
    centroids = get_random_centroids(k, x, y)
    
    for _ in range(max_iters):
        # Generate the cluster labels
        cluster_labels = assign_centroids(X, centroids)

        # Plot the graph
        plot(x, y, cluster_labels, centroids)

        # Store old centroids
        centroids_old = centroids
        
        # Update the centroids
        centroids = calculate_centroids(X, k, cluster_labels)

        # Check convergence
        if np.array_equal(centroids, centroids_old):
            break


if __name__ == "__main__":
    main()