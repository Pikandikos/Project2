#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import heapq
from data_reader import load_dataset

# Get arguments from command line and pass them to variables
def parse_args():
    parser = argparse.ArgumentParser(description="Neural LSH search")
    
    # Get all values from command line
    parser.add_argument("-d", required=True, help="Input dataset file")
    parser.add_argument("-q", required=True, help="Query file")
    parser.add_argument("-i", required=True, help="Index path prefix")
    parser.add_argument("-o", required=True, help="Output file")
    parser.add_argument("-type", required=True, choices=["sift", "mnist"])
    
    parser.add_argument("-N", type=int, default=1, help="Number of nearest neighbors")
    parser.add_argument("-R", type=float, default=None, help="Radius for range search")
    parser.add_argument("-T", type=int, default=5, help="Number of bins to probe")
    parser.add_argument("-range", type=str, default="true", choices=["true", "false"])
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    
    # Parse arguments for validation
    args = parser.parse_args()
    if args.N <= 0:
        parser.error(f" N must be positive, got {args.N}")
    if args.R is not None and args.R <= 0:
        parser.error(f" R must be positive if provided, got {args.R}")
    if args.T <= 0:
        parser.error(f" T must be positive, got {args.T}")
    if args.batch_size <= 0:
        parser.error(f" batch_size must be positive, got {args.batch_size}")

    return parser.parse_args()

# Euclidean distances between query and candidates.
def compute_distances(query, candidates):
    return np.linalg.norm(candidates - query, axis=1)

#Exact nearest neighbor search.
def exact_search(query, dataset, N=1):
    distances = compute_distances(query, dataset)
    
    # Get N nearest neighbors
    nearest_indices = np.argsort(distances)[:N]
    nearest_distances = distances[nearest_indices]

    return nearest_indices, nearest_distances

def search_single_query(query, model, dataset, inv_lists, T=5, N=1, R=None):
    # Search for a single query using Neural LSH.
    # Convert query to tensor for python functions
    query_tensor = torch.as_tensor(query, dtype=torch.float32).unsqueeze(0)
    
    # Get model predictions
    with torch.no_grad():
        scores = model(query_tensor)
        probabilities = torch.softmax(scores, dim=1)
    
    # Get top T bins
    top_t_bins = torch.topk(probabilities[0], T, largest=True, sorted=True).indices
    
    # Collect candidate indices
    top_t_bins_list = top_t_bins.cpu().tolist()
    candidate_indices = []
    for bin_idx in top_t_bins_list:
        candidate_indices.extend(inv_lists.get(bin_idx, [])) #if bin empty return empty list
    
    candidate_indices = list(set(candidate_indices))  # Remove duplicates
    
    if not candidate_indices:
        return [], [], []
    
    # Get candidate vectors
    candidates = dataset[candidate_indices]
    # Compute distances to candidates
    candidate_distances = compute_distances(query, candidates)
    
    # Check if we got enough candidates
    # Sort and get first N
    N_actual = min(N, len(candidate_distances))
    nearest_idx = np.argsort(candidate_distances)[:N_actual]
    nearest_indices = [candidate_indices[i] for i in nearest_idx]
    nearest_distances = candidate_distances[nearest_idx]
    
    # Get points within range
    id_in_range = []
    if R is not None and R > 0:
        id_in_range = [
            candidate_indices[i] 
            for i, dist in enumerate(candidate_distances) 
            if dist <= R
        ]
    
    return nearest_indices, nearest_candidate_dist, id_in_range
