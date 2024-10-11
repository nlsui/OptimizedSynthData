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
