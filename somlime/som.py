
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, output_size, input_size, sigma, learning_rate):
        # Initialize the SOM
        self.som = MiniSom(output_size[0], output_size[1], input_size, sigma=sigma, learning_rate=learning_rate, neighborhood_function='gaussian', random_seed=10)
        self.input_size = input_size
        self.output_size = output_size

    def train(self, data, epochs):
        # Train the SOM with the given data for a certain number of epochs
        self.som.train(data, epochs)

    def plot_distance_map(self):
        #scatterplot the winning points and superimpose a representation of the distance map
        plt.figure(figsize=(8, 7))
        plt.pcolor(self.som.distance_map().T, cmap='bone_r')
        plt.colorbar()
        plt.title('Distance map')
        plt.show()



    # Calculating distances to the centroids of the SOM clusters
    def distance_to_centroids(self, instance, perturbed_instances, exp):
        instance_centroid_position = self.som.winner(instance)

        # print("Instance centroid:",instance_centroid_position)
  
        pert_instance_centroids = [(perturbed, self.som.winner(perturbed)) for perturbed in perturbed_instances]
        # print("Perturbed instances centroids:", pert_instance_centroids)
        
        #Centroids distance     
        if (exp == 1):
            distances = [np.linalg.norm(np.array(instance_centroid_position,  dtype=np.float64) - np.array(pert_pos,  dtype=np.float64)) for pert_inst, pert_pos in pert_instance_centroids]
        elif (exp ==2):
        # Sum of  euclidean distance + centroid distance
            distances = [np.linalg.norm(np.array(instance,  dtype=np.float64) - np.array(pert_inst,  dtype=np.float64))+ np.linalg.norm(np.array(instance_centroid_position,  dtype=np.float64) - np.array(pert_pos,  dtype=np.float64)) for pert_inst, pert_pos in pert_instance_centroids]
       
       # Euclidean distances in the same cluster 
        elif (exp ==3): 
            distances=[]
            for pert_inst, pert_pos in pert_instance_centroids:
                if(pert_pos==instance_centroid_position):
                    distances.append(np.linalg.norm(np.array(instance,  dtype=np.float64) - np.array(pert_inst,  dtype=np.float64)))
                else:
                    distances.append(10.000)
        return np.array(distances)

