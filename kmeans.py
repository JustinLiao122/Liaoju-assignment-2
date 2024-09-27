import random
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation

class KMeans():



    def __init__(self, data, k , method ):
        self.data = data
        self.k = k
        self.method = method
        self.centroids = []
        self.clusters = {}
        self.history = []


    def which_method(self):
        if self.method == "Random":
            self.random_centroids()
        elif self.method == "Farthest First":
            self.farthest_first_centroids()
        elif self.method == "KMeans++":
            self.kmeans_plus_plus()


    def random_centroids(self):
        for i in range(0,self.k):
            random_point = random.choice(self. data)
            self.centroids.append(random_point)
        
    def farthest_first_centroids(self):
         self.centroids = [self.data[random.randint(0, len(self.data)-1)]]
         for _ in range(1, self.k):
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in self.centroids]) for x in self.data])
            probabilities = distances / distances.sum()
            cumulative_probs = np.cumsum(probabilities)
            r = random.random()
            next_centroid = self.data[np.searchsorted(cumulative_probs, r)]
            self.centroids.append(next_centroid)


    def kmeans_plus_plus(self):
        self.centroids = [self.data[random.randint(0, len(self.data)-1)]]
        for _ in range(1, self.k):
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in self.centroids]) for x in self.data])
            probabilities = distances / distances.sum()
            cumulative_probs = np.cumsum(probabilities)
            r = random.random()
            next_centroid = self.data[np.searchsorted(cumulative_probs, r)]
            self.centroids.append(next_centroid)
        print(len(self.centroids))
        

    def assign_clusters(self):
        self.clusters = {i: [] for i in range(self.k)}
        for node in self.data:
            index = self.closest_to(node)
            self.clusters[index].append(node)

        

    def closest_to(self, node):

        point1 = node
        min_distance = float('inf')
        index = -1
        for i in range(0,self.k):
            point2 = self.centroids[i]
            distance = math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))
            if min_distance >= distance:
                index = i 
                min_distance = distance

        return index

    def new_centroids(self):
       

        new_centroids = []
        for cluster in self.clusters.values():
            new_centroids.append(np.mean(cluster, axis=0))
        self.centroids = new_centroids




    def fit(self):
        
        self.which_method()
        old_centroids = self.centroids.copy()
        plt.ion()
        while(1):
            self.assign_clusters()
            self.history.append((self.centroids.copy(), {i: cluster[:] for i, cluster in self.clusters.items()}))
            self.new_centroids()
            if np.allclose(old_centroids, self.centroids):
                self.history.append((self.centroids.copy(), {i: cluster[:] for i, cluster in self.clusters.items()}))  # Capture final state
                break
            old_centroids = self.centroids.copy()
            plt.pause(0.5)  
        plt.ioff()



    def animate(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.get_cmap('tab10')  # Use a color map that can handle multiple categories

        def update(frame):
            ax.clear()
            centroids, clusters = self.history[frame]
            for i, cluster in clusters.items():
                if len(cluster) > 0:
                    cluster_points = np.array(cluster)
                    color = cmap(i % cmap.N)  # Get color from colormap, cycling if needed
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, label=f'Cluster {i}')
            centroids = np.array(centroids)
            ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=50, label='Centroids')
            ax.legend()
            ax.set_title(f"Iteration {frame + 1}")

        ani = FuncAnimation(fig, update, frames=len(self.history), repeat=False)
        plt.show()
        # Save animation as a video or gif
        ani.save('kmeans_animation.gif', writer='imagemagick', fps=1)  # or 'kmeans_animation.mp4'

    def reset(self):
        self.centroids = []
        self.clusters = {}
        self.history = []







def generate_random_points(num_points):
    x = np.random.random(num_points) * 20 - 10  # Scale to [-10, 10]
    y = np.random.random(num_points) * 20 - 10  # Scale to [-10, 10]
    points = np.column_stack((x, y)) # Combine x and y into a single array of points
    return points





# Example: Generate 100 random points
data = generate_random_points(200)
kmeans = KMeans(data, 6, "KMeans++")
kmeans.fit()
kmeans.animate()