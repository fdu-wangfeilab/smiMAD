import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, euclidean
from collections import defaultdict
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from libpysal.weights import KNN
from esda.moran import Moran

def morans_i(coords, preds):
    w = KNN(coords, k=5)
    moran = Moran(preds, w)
    return moran.I
    
def calculate_mmsccs(A, B, coords, labels, w_A=1/3, w_B=1/3, w_coords=1/3,n_components=10):
    """
    Calculates the Multi-Modal Spatial Cohesion and Separation score.

    Args:
        A (np.ndarray): N x dim1 feature matrix.
        B (np.ndarray): N x dim2 feature matrix.
        coords (np.ndarray): N x 2 coordinates matrix.
        labels (np.ndarray): N-dimensional vector of cluster labels.
        w_A, w_B, w_coords (float): Weights for each modality/space.

    Returns:
        float: MMSCS score (higher is better).
    """
    if len(np.unique(labels)) <= 1:
        return 0.0 # Cannot calculate separation for 1 cluster

    A = PCA(n_components=n_components,random_state=2025).fit_transform(A)
    B = PCA(n_components=n_components,random_state=2025).fit_transform(B)

    N = A.shape[0]
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    # 1. Normalize data
    scaler_A = StandardScaler()
    A_norm = scaler_A.fit_transform(A)
    scaler_B = StandardScaler()
    B_norm = scaler_B.fit_transform(B)
    scaler_coords = StandardScaler()
    coords_norm = scaler_coords.fit_transform(coords)

    centroids_A = np.zeros((num_clusters, A.shape[1]))
    centroids_B = np.zeros((num_clusters, B.shape[1]))
    centroids_coords = np.zeros((num_clusters, coords.shape[1]))
    intra_dist_A = np.zeros(num_clusters)
    intra_dist_B = np.zeros(num_clusters)
    intra_dist_coords = np.zeros(num_clusters)
    cluster_sizes = np.zeros(num_clusters, dtype=int)

    cluster_map = {label: i for i, label in enumerate(unique_labels)}
    points_in_cluster = defaultdict(list)
    for i, label in enumerate(labels):
        points_in_cluster[label].append(i)

    # 2 & 3. Calculate centroids and intra-cluster distances
    for label, idx in cluster_map.items():
        indices = points_in_cluster[label]
        cluster_sizes[idx] = len(indices)
        if cluster_sizes[idx] == 0: continue

        centroids_A[idx] = np.mean(A_norm[indices], axis=0)
        centroids_B[idx] = np.mean(B_norm[indices], axis=0)
        centroids_coords[idx] = np.mean(coords_norm[indices], axis=0)

        # Use cdist for potentially faster distance calculation
        intra_dist_A[idx] = np.mean(cdist(A_norm[indices], centroids_A[idx:idx+1]))
        intra_dist_B[idx] = np.mean(cdist(B_norm[indices], centroids_B[idx:idx+1]))
        intra_dist_coords[idx] = np.mean(cdist(coords_norm[indices], centroids_coords[idx:idx+1]))

    total_intra_dist = np.sum(
        (cluster_sizes / N) * (w_A * intra_dist_A + w_B * intra_dist_B + w_coords * intra_dist_coords)
    )

    # 4. Calculate inter-cluster distances
    if num_clusters > 1:
        inter_dist_A = np.mean(cdist(centroids_A, centroids_A))
        inter_dist_B = np.mean(cdist(centroids_B, centroids_B))
        inter_dist_coords = np.mean(cdist(centroids_coords, centroids_coords))
        # Note: cdist calculates all pairwise distances, including zeros on diagonal.
        # Mean of non-zero distances might be more appropriate, or use pdist.
        # For simplicity here, we use mean of the full cdist matrix, it's proportional.
        # A cleaner way: use pdist from scipy.spatial.distance
        from scipy.spatial.distance import pdist
        if num_clusters > 1:
             inter_dist_A = np.mean(pdist(centroids_A))
             inter_dist_B = np.mean(pdist(centroids_B))
             inter_dist_coords = np.mean(pdist(centroids_coords))
        else: # Should not happen due to initial check, but for safety
             inter_dist_A = inter_dist_B = inter_dist_coords = 0

        total_inter_dist = w_A * inter_dist_A + w_B * inter_dist_B + w_coords * inter_dist_coords
    else:
        total_inter_dist = 0


    # 5. Calculate MMSCS
    if total_intra_dist < 1e-9:  # Avoid division by zero or near-zero
        return (
            (np.inf if total_inter_dist > 0 else 0.0), 
            np.inf if inter_dist_A > 0 else 0.0,
            np.inf if inter_dist_B > 0 else 0.0,
            np.inf if inter_dist_coords > 0 else 0.0
        )

    mmsccs = total_inter_dist / total_intra_dist
    mmsccs_A = inter_dist_A / np.sum((cluster_sizes / N) * intra_dist_A)
    mmsccs_B = inter_dist_B / np.sum((cluster_sizes / N) * intra_dist_B)
    mmsccs_coords = inter_dist_coords / np.sum((cluster_sizes / N) * intra_dist_coords)

    return mmsccs


def calculate_nmmi(A, B, coords, labels, w_A=1/3, w_B=1/3, w_coords=1/3, n_components=10):
    """
    Calculates the Normalized Multi-Modal Intra-cluster Inertia score.
    Higher score (closer to 1) indicates better cohesion across modalities.
    """
    N = A.shape[0]
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    A = PCA(n_components=n_components,random_state=2025).fit_transform(A)
    B = PCA(n_components=n_components,random_state=2025).fit_transform(B)

    if num_clusters <= 1:
        return 0.0 # Inertia explained is 0 for 1 cluster

    # 1. Normalize data
    scaler_A = StandardScaler()
    A_norm = scaler_A.fit_transform(A)
    scaler_B = StandardScaler()
    B_norm = scaler_B.fit_transform(B)
    scaler_coords = StandardScaler()
    coords_norm = scaler_coords.fit_transform(coords)

    # 2. Calculate Total Sum of Squares (TSS) for each modality
    global_centroid_A = np.mean(A_norm, axis=0)
    global_centroid_B = np.mean(B_norm, axis=0)
    global_centroid_coords = np.mean(coords_norm, axis=0)

    # Use squared Euclidean distance for inertia calculation
    tss_A = np.sum(cdist(A_norm, global_centroid_A.reshape(1, -1), metric='sqeuclidean'))
    tss_B = np.sum(cdist(B_norm, global_centroid_B.reshape(1, -1), metric='sqeuclidean'))
    tss_coords = np.sum(cdist(coords_norm, global_centroid_coords.reshape(1, -1), metric='sqeuclidean'))

    # 3 & 4. Calculate Total Within-Cluster Sum of Squares (WSS)
    total_wss_A = 0.0
    total_wss_B = 0.0
    total_wss_coords = 0.0

    cluster_map = {label: i for i, label in enumerate(unique_labels)}
    points_in_cluster = defaultdict(list)
    for i, label in enumerate(labels):
        points_in_cluster[label].append(i)

    for label in unique_labels:
        indices = points_in_cluster[label]
        if len(indices) == 0: continue

        centroid_A_j = np.mean(A_norm[indices], axis=0)
        centroid_B_j = np.mean(B_norm[indices], axis=0)
        centroid_coords_j = np.mean(coords_norm[indices], axis=0)

        wss_A_j = np.sum(cdist(A_norm[indices], centroid_A_j.reshape(1, -1), metric='sqeuclidean'))
        wss_B_j = np.sum(cdist(B_norm[indices], centroid_B_j.reshape(1, -1), metric='sqeuclidean'))
        wss_coords_j = np.sum(cdist(coords_norm[indices], centroid_coords_j.reshape(1, -1), metric='sqeuclidean'))

        total_wss_A += wss_A_j
        total_wss_B += wss_B_j
        total_wss_coords += wss_coords_j

    # 5. Calculate Variance Explained Ratio for each modality
    r_A = 1.0 - (total_wss_A / tss_A) if tss_A > 1e-9 else 0.0
    r_B = 1.0 - (total_wss_B / tss_B) if tss_B > 1e-9 else 0.0
    r_coords = 1.0 - (total_wss_coords / tss_coords) if tss_coords > 1e-9 else 0.0

    # Ensure ratios are within [0, 1] due to potential float issues
    r_A = max(0.0, min(1.0, r_A))
    r_B = max(0.0, min(1.0, r_B))
    r_coords = max(0.0, min(1.0, r_coords))


    # 6. Calculate NMMI
    nmmi = w_A * r_A + w_B * r_B + w_coords * r_coords
    return nmmi
