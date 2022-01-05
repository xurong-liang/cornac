"""
Convert individual npz matrix file (each is a csr_matrix) back to np matrix
"""

import os
import argparse as ap
import scipy.sparse as sparse

matrices = ["X", "Y", "A", "U1", "U2", "V", "H1", "H2"]
parser = ap.ArgumentParser()
parser.add_argument("-path", type=str, required=True, help="The path to read individual npz file")

args = parser.parse_args()

for matrix in matrices:
    input_path = args.path + matrix + ".npz"
    if not os.path.isfile(input_path):
        print(f"Invalid input file path: {input_path}")
        exit(1)
    scr = sparse.load_npz(input_path)
    np_arr = scr.toarray()
    print(f"{matrix}: {np_arr.shape}")
