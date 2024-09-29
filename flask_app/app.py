import os
from flask import Flask, render_template, request, redirect, url_for , jsonify ,send_file
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation
import imageio

class KMeans():



    def __init__(self, data, k , method ,manuel):
        self.data = data
        self.k = k
        self.method = method
        self.centroids = []  
        if manuel:
            self.centroids = np.array(manuel)
        print(self.centroids)
        self.clusters = {}
        self.history = []


    def which_method(self):
        if len(self.centroids) > 0:
            return
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
        self.centroids = np.array(self.centroids)


    def farthest_first_centroids(self):
         self.centroids = [self.data[random.randint(0, len(self.data)-1)]]
         for _ in range(1, self.k):
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in self.centroids]) for x in self.data])
            probabilities = distances / distances.sum()
            cumulative_probs = np.cumsum(probabilities)
            r = random.random()
            next_centroid = self.data[np.searchsorted(cumulative_probs, r)]
            self.centroids = np.append(self.centroids, [next_centroid], axis=0)


    def kmeans_plus_plus(self):
        self.centroids = [self  .data[random.randint(0, len(self.data)-1)]]
        for _ in range(1, self.k):
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in self.centroids]) for x in self.data])
            probabilities = distances / distances.sum()
            cumulative_probs = np.cumsum(probabilities)
            r = random.random()
            next_centroid = self.data[np.searchsorted(cumulative_probs, r)]
            self.centroids = np.append(self.centroids, [next_centroid], axis=0)
        
        

    def assign_clusters(self):
        self.clusters = {i: [] for i in range(self.k)}  
        for node in self.data:
            index = self.closest_to(node)  
            self.clusters[index].append(node)  

    def closest_to(self, node):
            distances = [np.linalg.norm(node - centroid) for centroid in self.centroids]
            min_distance = np.min(distances)
    
            closest_indices = [i for i, dist in enumerate(distances) if dist == min_distance]
    
            if len(closest_indices) > 1: 
                return min(closest_indices, key=lambda i: len(self.clusters[i]))
    
            return np.argmin(distances) 


    def new_centroids(self):

        new_centroids = []
        for cluster in self.clusters.values():
            new_centroids.append(np.mean(cluster, axis=0))
        else:
             print(f"Cluster {cluster} has no points, reinitializing centroid.")
        self.centroids = np.array(new_centroids)




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
        return True



    def animate(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.get_cmap('tab10')  

        def update(frame):
            ax.clear()
            centroids, clusters = self.history[frame]
            for i, cluster in clusters.items():
                if len(cluster) > 0:
                    cluster_points = np.array(cluster)
                    color = cmap(i % cmap.N)  
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, label=f'Cluster {i}')
            centroids = np.array(centroids)
            ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=50, label='Centroids')
            
            ax.set_title(f"Iteration {frame + 1}")

        ani = FuncAnimation(fig, update, frames=len(self.history), repeat=False)
        plt.show()
        
        ani.save('static/kmeans_animation.gif', writer='imagemagick', fps=1) 

        plt.close(fig) 
        return True
    




    def reset(self):
        self.centroids = []
        self.clusters = {}
        self.history = []

    def get_current_history(self, frame_index):
        return self.history[frame_index] if 0 <= frame_index < len(self.history) else None


def start_graph(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    all_points = data  
    ax.scatter(all_points[:, 0], all_points[:, 1], c='blue', label='Data Points')
    ax.set_title("Initial Data Set")
    ax.legend()

    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            selected_point = (event.xdata, event.ydata)
            
            print(f"Selected point: {selected_point}")
          
    cid = fig.canvas.mpl_connect('button_press_event', on_click)  
    plt.savefig('static/kmeans_animation.png', format='png', dpi=100)
    plt.close(fig)

    image = imageio.imread('static/kmeans_animation.png')
    imageio.imwrite('static/kmeans_animation.gif', image)
    return True



def generate_random_points(num_points):
    x = np.random.random(num_points) * 20 - 10  
    y = np.random.random(num_points) * 20 - 10  
    points = np.column_stack((x, y))
    return points















app = Flask(__name__)
app.secret_key = os.urandom(24)

k_value = -1
method = 'Random'
data_points = []
manual_centroids = []
kmeans = None
index = 0

@app.route('/')
def index():
    global k_value , data_points , manual_centroids
    k_value = -1
    manual_centroids = []
    data_points = generate_random_points(200)
    index = 0
    start_graph(data_points)
    return render_template('index.html')

@app.route('/set_number', methods=['POST'])
def set_k():
    global k_value , kmeans , index
    data = request.get_json()
    k_value = int(data['k_value'])
    kmeans = None 
    index = 0 
    return jsonify(success=True) 


@app.route('/set_method', methods = ['POST'])
def set_method():
    global method , manual_centroids ,kmeans,index
    data = request.get_json()  
    method = data['method']  
    if method != 'Manual':
        manual_centroids = []
    kmeans = None
    index = 0
    return jsonify(method=method)



@app.route('/step_through', methods=['POST'])
def run_kmeans():
    global k_value, method, data_points, manual_centroids ,kmeans, index

    if method == "Manual" and len(manual_centroids) == 0:
        return jsonify(success=True, message="Plz select centriods", animation_url='/get_animation')

    if kmeans is None:
        run()
        index = 0

    if index < len(kmeans.history):
        last_centroids, last_clusters = kmeans.history[index] 

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.get_cmap('tab10')

        ax.clear()
        for i, cluster in last_clusters.items():
            if len(cluster) > 0:
                cluster_points = np.array(cluster)
                color = cmap(i % cmap.N)
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, label=f'Cluster {i}')
        last_centroids = np.array(last_centroids)
        ax.scatter(last_centroids[:, 0], last_centroids[:, 1], c='red', marker='x', s=75, linewidths=3,label='Centroids')
        ax.set_title("Iteration " + str(index))

        animation_file = 'static/kmeans_animation.gif'
        plt.savefig(animation_file, format='png', dpi=100) 
        plt.close(fig)  
        index = index +1

    if index == len(kmeans.history):
        return jsonify(success=True, message="Run to convergence complete", animation_url='/get_animation')
   
    return jsonify(success=True, animation_url = '/get_animation')


@app.route('/get_animation', methods=['GET'])
def get_animation():
    animation_file = 'static/kmeans_animation.gif'  
    return send_file(animation_file, mimetype='image/gif')



@app.route('/add_centroid', methods=['POST'])
def add_centroid():
    global manual_centroids ,k
    data = request.get_json()
    x, y = data['x'], data['y']
    manual_centroids.append([x, y])
    
    print(manual_centroids)
    return jsonify(success=True)

@app.route('/get_k_value', methods=['GET'])
def get_k_value():
    global k_value
    return jsonify(k_value=k_value)


@app.route('/get_method', methods=['GET'])
def get_method():
    global method
    return jsonify(method=method)


@app.route('/newData', methods= ['POST'])
def generate_new_data():
    global  data_points , manual_centroids , kmeans , index ,manual_centroids
    manual_centroids = []
    data_points = generate_random_points(200)
    start_graph(data_points)

    kmeans = None
    animation_url = 'static/kmeans_animation.gif'
    index = 0 
    manual_centroids = []
    return jsonify(success=True, animation_url=animation_url)



@app.route('/resetAlg' ,methods = ['POST'])
def reset_Alg():
    global data_points ,kmeans, index ,manual_centroids

    start_graph(data_points)

    animation_url = 'static/kmeans_animation.gif'
    kmeans = None
    index = 0 
    manual_centroids = []
    return jsonify(success=True , animation_url = animation_url) 




@app.route('/runtoConvergance' , methods =['POST'])
def runtoConvergance():
    global kmeans
    if kmeans is None:
        run()
    if kmeans.history:  
        last_centroids, last_clusters = kmeans.history[-1]  

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.get_cmap('tab10')

        ax.clear()
        for i, cluster in last_clusters.items():
            if len(cluster) > 0:
                cluster_points = np.array(cluster)
                color = cmap(i % cmap.N)
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, label=f'Cluster {i}')
        last_centroids = np.array(last_centroids)
        ax.scatter(last_centroids[:, 0], last_centroids[:, 1], c='red', marker='x', s=75,linewidths=3, label='Centroids')
        ax.set_title("Final Iteration")

        animation_file = 'static/kmeans_animation.gif'
        plt.savefig(animation_file, format='png', dpi=100) 
        plt.close(fig)  

        return jsonify(success=True, animation_url='/get_animation')
    else:
        return jsonify(success=False, message="No iterations recorded"), 400
        


def run():
    global k_value, method, data_points, manual_centroids ,kmeans

    print(manual_centroids)
    kmeans = KMeans(data_points ,k_value, method , manual_centroids)
    kmeans.fit()  
    
    return 

if __name__ == '__main__':
    app.run(debug=True)









