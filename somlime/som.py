
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
    def distance_to_centroids(self, instance, perturbed_instances):
        instance_centroid_position = self.som.winner(instance)

        print("Instance centroid:",instance_centroid_position)
  
        pert_instance_centroids = [(perturbed, self.som.winner(perturbed)) for perturbed in perturbed_instances]
        print("Perturbed instances centroids:", pert_instance_centroids)
    
    
        same_cluster_inst_pos= []
 
        for pert_inst, pert_pos in pert_instance_centroids:
            if instance_centroid_position == pert_pos:
                same_cluster_inst_pos.append(pert_inst)
                print("\nInstances in the same cluster", pert_inst, pert_pos)
            

        distances_in_cluster = [np.linalg.norm(np.array(instance,  dtype=np.float64) - np.array(pert_inst,  dtype=np.float64)) for pert_inst in same_cluster_inst_pos]
        
        # Sum of centroid distance + euclidean distance 
        distances_out_of_cluster = [np.linalg.norm(np.array(instance,  dtype=np.float64) - np.array(pert_inst,  dtype=np.float64))+ np.linalg.norm(np.array(instance_centroid_position,  dtype=np.float64) - np.array(pert_pos,  dtype=np.float64)) for pert_inst, pert_pos in pert_instance_centroids]
        return list(zip(same_cluster_inst_pos, distances_in_cluster))




if __name__ == "__main__":
    def generate_perturbed_instances(original_instance, epsilon, num_instances=10):
        perturbed_instances = []
        for _ in range(num_instances):
            perturbation = epsilon * np.random.normal(size=original_instance.shape)
            perturbed_instance = original_instance + perturbation
            perturbed_instance = np.clip(perturbed_instance, 0, 1)

            perturbed_instances.append(perturbed_instance)
        return perturbed_instances


    epsilon = 0.2
    num_instances = 10
    instance_to_explain = np.random.rand(5)
    perturbed_instances = generate_perturbed_instances(instance_to_explain, epsilon, num_instances)
    input_size = 5
    output_size = (5,5)
    data = np.random.rand(100, 5)
    labels=[]
    epochs =1000

    som = SOM(output_size, input_size, sigma=0.1, learning_rate=0.1)
    som.train(data, epochs)



    distances = som.distance_to_centroids(instance_to_explain, perturbed_instances)


    if(len(distances)!= 0):
        best_perturbed_data = min(distances, key=lambda x: x[1])
        print("Instance to explain:", instance_to_explain)
        print("Best perturbed data:", best_perturbed_data)
    else:
        print("No instance in the same cluster")