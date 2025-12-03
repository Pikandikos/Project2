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

# Import MLPClassifier from build.py
import sys
import os

# Add the parent directory to Python path to import from build.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlsh_build import MLPClassifier  # Now you can import it


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

    return args

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

# Search for a single query using Neural LSH.
def search_single_query(query, model, dataset, inv_lists, T=5, N=1, R=None):
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
    nearest_candidate_dist = candidate_distances[nearest_idx]
    
    # Get points within range
    id_in_range = []
    if R is not None and R > 0:
        id_in_range = [
            candidate_indices[i] 
            for i, dist in enumerate(candidate_distances) 
            if dist <= R
        ]
    
    return nearest_indices, nearest_candidate_dist, id_in_range

def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set radius default based on dataset type
    if args.R is None:
        args.R = 2000.0 if args.type == "mnist" else 2800.0
    
    # Load datasets
    print(f"Loading dataset from {args.d}")
    dataset = load_dataset(args.d, args.type)
    print(f"Loading queries from {args.q}")
    queries = load_dataset(args.q, args.type)
    
    # Load model and index
    model_path = args.i + "_model.pth"
    index_path = args.i + "_index.pkl"
    
    print(f"Loading model from {model_path}")

    model = torch.load(model_path, weights_only=False)
    
    print(f"Loading index from {index_path}")
    with open(index_path, "rb") as f:
        index_data = pickle.load(f)
    
    # pass file contect into variables
    inv_lists = index_data["inv_lists"]
    m = index_data["m"]
    
    # Prepare output
    output_lines = ["Neural LSH"]
    
    # Metrics
    total_approx_time = 0
    total_exact_time = 0
    total_recall = 0
    total_af = 0
    total_queries = 0
    
    # Process each query
    for query_idx, query in enumerate(queries):
        output_lines.append(f"Query: {query_idx}")
        
        # Exact search
        exact_start = time.time()
        true_nearest_indices, true_nearest_distances= exact_search(query, dataset, N=args.N)
        exact_end = time.time()
        exact_time = exact_end - exact_start
        
        # Approximate search (Neural LSH)
        approx_start = time.time()
        approx_nearest_indices, approx_distances, approx_in_range = search_single_query(
            query, model, dataset, inv_lists, 
            T=args.T, N=args.N, 
            R=args.R if args.range == "true" else None)
        approx_end = time.time()
        approx_time = approx_end - approx_start
        
        # Output approximate results
        for i, (idx, dist) in enumerate(zip(approx_nearest_indices[:args.N], approx_distances[:args.N])):
            if i < len(true_nearest_indices):
                true_dist = true_nearest_distances[i]
                output_lines.append(f"Nearest neighbor-{i+1}: {idx}")
                output_lines.append(f"distanceApproximate: {dist:.6f}")
                output_lines.append(f"distanceTrue: {true_dist:.6f}")
                
                # Compute approximation factor (avoid division by zero)
                if true_dist > 1e-10:
                    af = dist / true_dist
                else:
                    af = 1.0
                total_af += af
            else:
                output_lines.append(f"Nearest neighbor-{i+1}: {idx}")
                output_lines.append(f"distanceApproximate: {dist:.6f}")
                output_lines.append(f"distanceTrue: -")
        
        # Output R-near neighbors if range search enabled
        if args.range == "true":
            output_lines.append("R-near neighbors:")
            for idx in approx_in_range:
                output_lines.append(str(idx))
        
        # Compute recall
        recall = 0
        true_set = set(true_nearest_indices[:args.N])
        approx_set = set(approx_nearest_indices[:args.N])
        recall = len(true_set.intersection(approx_set)) / args.N
        
        total_recall += recall
        total_approx_time += approx_time
        total_exact_time += exact_time
        total_queries += 1
    
    # Compute final metrics
    if total_queries > 0:
        avg_af = total_af / total_queries
        avg_recall = total_recall / total_queries
        avg_approx_time = total_approx_time / total_queries
        avg_exact_time = total_exact_time / total_queries
        qps = total_queries / total_approx_time if total_approx_time > 0 else 0
        
        output_lines.append(f"Average AF: {avg_af:.6f}")
        output_lines.append(f"Recall@{args.N}: {avg_recall:.6f}")
        output_lines.append(f"QPS: {qps:.6f}")
        output_lines.append(f"tApproximateAverage: {avg_approx_time:.6f}")
        output_lines.append(f"tTrueAverage: {avg_exact_time:.6f}")
    
    # Write output
    with open(args.o, 'w') as f:
        f.write("\n".join(output_lines))
    
    print(f"Search completed. Results written to {args.o}")
    print(f"Average Recall@{args.N}: {avg_recall:.4f}")
    print(f"Average AF: {avg_af:.4f}")
    print(f"QPS: {qps:.2f}")

if __name__ == "__main__":
    main()
