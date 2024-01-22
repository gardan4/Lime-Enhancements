import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class SOM:
    def __init__(self, output_size, input_size, sigma, learning_rate):
        # Initialize the SOM
        self.som = MiniSom(output_size[0], output_size[1], input_size, sigma=sigma, learning_rate=learning_rate,
                           neighborhood_function='gaussian', random_seed=10)
        self.input_size = input_size
        self.output_size = output_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def train(self, data, epochs):
        # Train the SOM with the given data for a certain number of epochs
        self.som.train(data, epochs)

    def plot_distanceAndDensity_map(self, pert_instance_centroids, instance_centroid_position):
        density_map = np.zeros((self.output_size[0], self.output_size[1]))
        # Count instances in each cell of the grid
        for t in pert_instance_centroids:
            x, y = t[1]
            density_map[x, y] += 1

        fig, (ax0, ax1) = plt.subplots(1, 2)

        # Plot distance map
        ax0.pcolor(self.som.distance_map().T, cmap='bone_r')
        ax0.set_title('Distance map, sigma: ' + str(self.sigma) + ", learning rate: " + str(self.learning_rate))

        # Plot density map
        ax1.pcolor(density_map.T, cmap='bone_r')
        ax1.set_title('Density map, sigma: ' + str(self.sigma) + ", learning rate: " + str(self.learning_rate))

        # Adjust layout and colorbars
        fig.tight_layout()
        fig.colorbar(ax0.pcolor(self.som.distance_map().T, cmap='bone_r'), ax=ax0)
        fig.colorbar(ax1.pcolor(density_map.T, cmap='bone_r'), ax=ax1)
        fig.subplots_adjust(wspace=0.5)

        # Ensure coordinates are within the range and visible
        scatter_x = float(instance_centroid_position[0] + 0.5)
        scatter_y = float(instance_centroid_position[1] + 0.5)
        ax1.scatter(scatter_x, scatter_y, color='red', marker='o', s=50, label='Circle Center')  # s is the size

        plt.show()


    # Calculating distances to the centroids of the SOM clusters
    def distance_to_centroids(self, instance, perturbed_instances, scaler_inv, exp, plot=False):
        instance_centroid_position = self.som.winner(instance)
        # print("Instance", instance)
        # print("Instance centroid:",instance_centroid_position)

        instance_centroid_position_2 = self.som.winner(instance)

        # print("Instance centroid:",instance_centroid_position_2)

        pert_instance_centroids = [(perturbed, self.som.winner(perturbed)) for perturbed in perturbed_instances]
        # print("Perturbed instances centroids: ", *pert_instance_centroids, sep='\n')

        for pert_inst, pert_pos in pert_instance_centroids:
            print("Perturbed instances centroids: ", scaler_inv.inverse_transform(pert_inst.reshape(1, -1)), pert_pos)

        # Centroids distance
        if (exp == 1):
            distances = [np.linalg.norm(
                np.array(instance_centroid_position, dtype=np.float64) - np.array(pert_pos, dtype=np.float64)) for
                pert_inst, pert_pos in pert_instance_centroids]
        elif (exp == 2):
            # Sum of  euclidean distance + centroid distance
            distances = [np.linalg.norm(
                np.array(instance, dtype=np.float64) - np.array(pert_inst, dtype=np.float64)) + np.linalg.norm(
                np.array(instance_centroid_position, dtype=np.float64) - np.array(pert_pos, dtype=np.float64)) for
                         pert_inst, pert_pos in pert_instance_centroids]
        # Euclidean distances in the same cluster
        elif (exp == 3):
            distances = []
            for pert_inst, pert_pos in pert_instance_centroids:
                if (pert_pos == instance_centroid_position):
                    distances.append(
                        np.linalg.norm(np.array(instance, dtype=np.float64) - np.array(pert_inst, dtype=np.float64)))
                else:
                    distances.append(10.000)

        # Plot the distance map
        if plot:
            self.plot_distanceAndDensity_map(pert_instance_centroids, instance_centroid_position)

        return np.array(distances)
