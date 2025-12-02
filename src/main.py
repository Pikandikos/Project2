from nlsh_build import *
from data_reader import *

def main():
    args = parse_args()
    np.random.seed(args.seed)

    print(f"Loading dataset from {args.d} (type={args.type})")
    dataset = load_dataset(args.d, args.type)  # Done in data_reader.py
    #
    dataset = dataset[:5000]
    n, d = dataset.shape
    print(f"Loaded {n} points of dimension {d}")

    print(f"Building k-NN graph with k={args.knn}")
    neighbors = build_knn_graph(dataset, k=args.knn)

    print("Building weighted edges")
    edge_dict = build_weighted_edges(neighbors)

    print("Converting to CSR format for KaHIP")
    xadj, adjncy, adjcwgt, vwgt = edges_to_csr(n, edge_dict)

    print(f"Running KaHIP with m={args.m}, imbalance={args.imbalance}, mode={args.kahip_mode}")
    labels = run_kahip(xadj, adjncy, adjcwgt, vwgt,
                       m=args.m, imbalance=args.imbalance,
                       mode=args.kahip_mode, seed=args.seed)
    print("KaHIP done.")

    print("Training MLP classifier")
    model = train_mlp(dataset, labels,
                      m=args.m,
                      layers=args.layers,
                      nodes=args.nodes,
                      epochs=args.epochs,
                      batch_size=args.batch_size,
                      lr=args.lr,
                      seed=args.seed)

    print("Building inverted lists")
    inv_lists = build_inverted_lists(labels, m=args.m)

    # saving
    os.makedirs(os.path.dirname(args.i), exist_ok=True) if os.path.dirname(args.i) else None
    model_path = args.i + "_model.pth"
    index_path = args.i + "_index.pkl"

    print(f"Saving model to {model_path}")
    torch.save(model, model_path)

    print(f"Saving index to {index_path}")
    with open(index_path, "wb") as f:
        pickle.dump(
            {
                "points": dataset.astype(np.float32),
                "labels": labels,
                "inv_lists": inv_lists,
                "m": args.m,
            },
            f,
        )

    print("nlsh_build finished.")


if __name__ == "__main__":
    main()

