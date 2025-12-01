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
