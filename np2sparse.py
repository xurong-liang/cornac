"""
Convert numpy matrices in matrices.py to csr_matrix, each stored in an individual
npz file
"""
import os
import argparse as ap
import scipy.sparse as sparse
import numpy as np

matrices = ["X", "Y", "A", "U1", "U2", "V", "H1", "H2"]
parser = ap.ArgumentParser()
parser.add_argument("-path", type=str, required=True, help="The path to read matrices.npy and"
                                                           "output individual npz file")
args = parser.parse_args()

input_path = args.path + "matrices.npy"

if not os.path.isfile(input_path):
    print(f"Invalid input file path: {input_path}")
    exit(1)

with open(input_path, "rb") as fp:
    for matrix in matrices:
        out = np.load(fp)
        out = sparse.csr_matrix(out)
        with open(args.path + f"{matrix}.npz", "wb") as write_head:
            sparse.save_npz(write_head, out)
print("done")
