import numpy as np

import numpy as np
import struct


# ---------- MNIST IMAGES (Python version of read_mnist_im) ----------
def read_mnist_im(path: str) -> np.ndarray:
    """
    MNIST image file format (big-endian):
      int32 magic_number
      int32 number_of_images
      int32 n_rows
      int32 n_cols
      then number_of_images * n_rows * n_cols bytes (unsigned char pixels)
    """
    with open(path, "rb") as f:
        # Read big-endian int32s
        magic_number = np.fromfile(f, dtype=">i4", count=1)[0]
        number_of_images = np.fromfile(f, dtype=">i4", count=1)[0]
        n_rows = np.fromfile(f, dtype=">i4", count=1)[0]
        n_cols = np.fromfile(f, dtype=">i4", count=1)[0]

        # Read pixel data as unsigned bytes
        num_pixels = number_of_images * n_rows * n_cols
        data = np.fromfile(f, dtype=np.uint8, count=num_pixels)
        if data.size != num_pixels:
            raise ValueError(
                f"MNIST file {path} truncated: expected {num_pixels} bytes, got {data.size}"
            )

    # Reshape to (number_of_images, n_rows * n_cols) and convert to float32
    images = data.reshape(number_of_images, n_rows * n_cols).astype(np.float32)
    return images


# ---------- SIFT VECTORS (Python version of read_sift) ----------
def read_sift(path: str) -> np.ndarray:
    """
    File format (little-endian):
      Repeated until EOF:
        int32 dimension           // usually 128
        float32[dimension] vector // little-endian
    """
    vectors = []

    with open(path, "rb") as f:
        while True:
            # Read 4 bytes for dimension
            dim_bytes = f.read(4)
            if not dim_bytes:
                break  # EOF cleanly

            if len(dim_bytes) < 4:
                # Incomplete header at end of file
                break

            # Little-endian int32 for dimension
            (dimension,) = struct.unpack("<i", dim_bytes)

            # Read 'dimension' float32 values (little-endian)
            vec = np.fromfile(f, dtype="<f4", count=dimension)
            if vec.size != dimension:
                # Incomplete vector at end of file
                break

            # Optional: warn if dimension != 128 (like your C++)
            if dimension != 128:
                print(f"Warning: Unexpected dimension {dimension} (expected 128)")

            vectors.append(vec.astype(np.float32))

    if not vectors:
        raise ValueError(f"No SIFT vectors read from {path}")

    X = np.vstack(vectors)  # shape = (n, 128)
    return X


# ---------- Combined loader (this is what main() calls) ----------
def load_dataset(path: str, dtype: str) -> np.ndarray:
    """
    dtype:
      - "mnist": use MNIST image format (idx3-ubyte style)
      - "sift" : use SIFT binary format from your C++ read_sift
    Returns:
      X: np.ndarray of shape (n, d), float32
    """
    dtype = dtype.lower()
    if dtype == "mnist":
        return read_mnist_im(path)
    elif dtype == "sift":
        return read_sift(path)
    else:
        raise ValueError(f"Unknown dtype '{dtype}', expected 'mnist' or 'sift'")

