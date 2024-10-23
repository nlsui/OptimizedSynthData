import numpy as np
from scipy.spatial.distance import cdist


def nearest_neighbor_distances(embeddings):
    """
    Calculates the Euclidean distance to the nearest neighbor for each embedding in the array.

    Parameters:
    embeddings (array-like): A list or array of embedding vectors.

    Returns:
    distances (list): A list of distances to the nearest neighbor for each embedding.
    """
    # Convert embeddings to a NumPy array (if not already)
    embeddings = np.array(embeddings)

    # Calculate the pairwise distances between all embeddings
    pairwise_distances = cdist(embeddings, embeddings, metric='euclidean')

    # For each embedding, find the distance to the nearest neighbor (excluding itself)
    nearest_distances = []
    for i, distances in enumerate(pairwise_distances):
        # Exclude the distance to itself (which is always 0)
        nearest_distance = np.min(distances[np.nonzero(distances)])
        nearest_distances.append(nearest_distance)

    return nearest_distances


def calculate_threshold(embeddings, std_multiplier=3):
    """
    Calculates the threshold based on mean ± std_multiplier * standard deviation.

    Parameters:
    embeddings (array-like): A list or array of embedding vectors.
    std_multiplier (float): Multiplier for standard deviation (default is 3).

    Returns:
    lower_threshold (float): The lower bound for normal nearest neighbor distances.
    upper_threshold (float): The upper bound for normal nearest neighbor distances.
    """
    # Calculate nearest neighbor distances
    embeddings = np.array(embeddings)
    pairwise_distances = cdist(embeddings, embeddings, metric='euclidean')
    nearest_distances = [np.min(dist[np.nonzero(dist)]) for dist in pairwise_distances]

    # Calculate mean and standard deviation
    mean_distance = np.mean(nearest_distances)
    std_distance = np.std(nearest_distances)

    # Calculate the lower and upper threshold
    lower_threshold = mean_distance - std_multiplier * std_distance
    upper_threshold = mean_distance + std_multiplier * std_distance

    return lower_threshold, upper_threshold


def classify_embeddings(data, lower_threshold, upper_threshold):
    """
    Classifies embeddings based on nearest neighbor distances relative to the thresholds.

    Parameters:
    data (list of dicts): List of dictionaries, each containing 'text' and 'embedding'.
                          Example: [{'text': ..., 'embedding': ...}, ...]
    lower_threshold (float): The lower bound for normal nearest neighbor distances.
    upper_threshold (float): The upper bound for normal nearest neighbor distances.

    Returns:
    within_threshold (list): Embeddings within the thresholds.
    below_threshold (list): Embeddings with nearest neighbor distances below the lower threshold.
    above_threshold (list): Embeddings with nearest neighbor distances above the upper threshold.
    """
    # Prepare containers for different categories
    within_threshold = []
    below_threshold = []
    above_threshold = []

    # Extract embeddings
    embeddings = [item.embedding for item in data]

    # Calculate nearest neighbor distances
    pairwise_distances = cdist(embeddings, embeddings, metric='euclidean')
    nearest_distances = [np.min(dist[np.nonzero(dist)]) for dist in pairwise_distances]

    # Classify based on thresholds
    for i, dist in enumerate(nearest_distances):
        if dist < lower_threshold:
            below_threshold.append(data[i])
        elif dist > upper_threshold:
            above_threshold.append(data[i])
        else:
            within_threshold.append(data[i])

    return within_threshold, below_threshold, above_threshold
