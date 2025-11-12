import kmedoids
import pyclustering.cluster.clarans as clarans
import sys
import numpy as np
import os
import csv



def save_results_to_csv(results, filename):
    """
    Saves the clustering results to a CSV file.
    Parameters:
    - Instance
    - Number of points
    - Number of clusters
    - Method used
    - Execution time
    - Cost
    - Cost per point
    - results: A list with the medoids
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Instance', 'Num_Points', 'Num_Clusters', 'Method', 'Execution_Time', 'Cost', 'Cost_Per_Point', 'Medoids'])
        for result in results:
            writer.writerow(result)

def read_input_file(file_path):
    """
    Reads a file containing every point's coordinates (separated by ;) and returns a list of tuples.
    The floats are represented with a comma as decimal separator.
    """
    points = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            point = tuple(map(lambda x: float(x.replace(',', '.')), row))
            points.append(point)
    return points


def padronizar_como_R(X):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, ddof=1, keepdims=True)
    sigma[sigma == 0.0] = 1.0
    return (X - mu) / sigma


def distance_matrix_from_points(points):
    """
    Computes the distance matrix from a list of points.

    Parameters:
    - points: A list of tuples representing the coordinates of the points.

    Returns:
    - A 2D numpy array representing the distance matrix.
    """
    num_points = len(points)
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            distance_matrix[i][j] = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
    return distance_matrix


def run_clarans_experiment(points, num_medoids, max_iter=2):
    """
    Runs the CLARANS clustering algorithm on the given points.

    Parameters:
    - points: A list of tuples representing the coordinates of the points.
    - num_medoids: The number of medoids to use for clustering.
    - max_iter: Maximum number of iterations for the CLARANS algorithm.

    Returns:
    - medoids: Indices of the selected medoids.
    - clusters: A list where each element is a list of indices belonging to that cluster.
    """

    num_points = len(points)
    clarans_instance = clarans.clarans(points, num_medoids, max_iter, np.sqrt(num_points))
    print(f"Running CLARANS with {num_medoids} medoids and max_iter={max_iter} on {num_points} points.")
    timer_start = os.times()[4]
    clarans_instance.process()
    timer_end = os.times()[4]
    execution_time = timer_end - timer_start

    medoids = clarans_instance.get_medoids()
    clusters = clarans_instance.get_clusters()

    return medoids, execution_time, clusters


def run_experiment(distance_matrix, num_medoids, max_iter=100, method='pam'):
    """
    Runs the k-medoids clustering algorithm on the given distance matrix.

    Parameters:
    - distance_matrix: A 2D numpy array representing the distance matrix.
    - num_medoids: The number of medoids to use for clustering.
    - max_iter: Maximum number of iterations for the k-medoids algorithm.

    Returns:
    - medoids: Indices of the selected medoids.
    - clusters: A list where each element is a list of indices belonging to that cluster.
    """


    # Initialize k-medoids
    kmedoids_instance = kmedoids.KMedoids(n_clusters=num_medoids, max_iter=max_iter, method=method, random_state=42, init='build')
    
    # Fit the model
    timer_start = os.times()[4]
    kmedoids_instance.fit(distance_matrix)
    timer_end = os.times()[4]
    execution_time = timer_end - timer_start
    
    medoids = kmedoids_instance.medoid_indices_
    medoids.sort()
    
    cost = kmedoids_instance.inertia_

    return medoids, execution_time, cost


instances_names = ["Concrete_Data.i",
                    "DOWJONES.I",
                    "Prima_Indians.i",
                    "SPRDESP.I",
                    "TRIPADVISOR.I",
                    "Uniform700.i",
                    "banknote_authentication.I",
                    "broken-ring.i",
                    "chart.i",
                    "ecoli.i",
                    "gauss9.i",
                    "haberman.i",
                    "indian.i",
                    "numbers2.i",
                    "synthetic_control.i",
                    "wdbc.I",
                    "yeast.i",
                    "waveform.I"
                ]

methods = [
           "fasterpam",
           "fastpam1",
           "pam",
           "alternate"
           #"clarans"
           ]

number_of_clusters = [3, 4, 5, 6, 7, 20, 25]

if __name__ == "__main__":
    results = []
    for instance_name in instances_names:
        points = read_input_file(f"../../../instances/general/{instance_name}")
        num_points = len(points)

        points = padronizar_como_R(np.array(points))

        if "clarans" in methods:
            for num_clusters in number_of_clusters:
                medoids, execution_time, cost = run_clarans_experiment(points, num_clusters)
                results.append([instance_name, num_points, num_clusters, "clarans", execution_time, cost, medoids])
                print(f"Instance: {instance_name}, Points: {num_points}, Clusters: {num_clusters}, Method: clarans, Time: {execution_time:.4f}, Cost: {cost}")
            methods.remove("clarans")
        
        distance_matrix = distance_matrix_from_points(points)

        for num_clusters in number_of_clusters:
            for method in methods:
                medoids, execution_time, cost = run_experiment(distance_matrix, num_clusters, method=method)
                results.append([instance_name, num_points, num_clusters, method, execution_time, cost, cost/num_points, medoids])
                print(f"Instance: {instance_name}, Points: {num_points}, Clusters: {num_clusters}, Method: {method}, Time: {execution_time:.4f}, Cost: {cost}")
    save_results_to_csv(results, "../results/clustering_results_clarans.csv")