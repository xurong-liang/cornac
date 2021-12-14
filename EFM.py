# perform EFM on a dataset
# requires: 1. reviews.pickle, as per obtained from 7. matched.py (Provide the
# path to the file)
#           2. The directory where the experiment results can be saved

import cornac
import os
import argparse as ap

import numpy as np
from cornac.data import SentimentModality
from cornac.eval_methods import RatioSplit
from common_func import read_file, get_ratings_and_sentiments, test_ratio, validate_ratio


parser = ap.ArgumentParser()
parser.add_argument("-input", type=str, required=True, help="The location of reviews.pickle")
parser.add_argument("-output", type=str, default="None", help="Dir to output directory; "
                                                              "if no need then type 'None'")
parser.add_argument("-save_matrices", type=str, default="yes", help="whether to save X, Y, U1,"
                                                                    " U2, H1, H2, V matrices "
                                                                    "to the output dir")
# r
parser.add_argument("-r", type=int, required=True, help="The number of explicit factors")
# r'
parser.add_argument("-r_prime", type=int, required=True, help="The number of implicit factors")
parser.add_argument("-name", type=str, required=True, help="Name of the dataset")
args = parser.parse_args()

seed_num = 1

path, res_path, save_matrices, dataset_name = \
    args.input, args.output, args.save_matrices.lower(), args.name

save_matrices = True if save_matrices == "yes" else False

if save_matrices and args.output == "None":
    print("Please specify the output folder")
    exit(1)

if not os.path.isfile(path):
    print("Invalid input file path")
    exit(1)
if not os.path.isdir(res_path):
    print("Invalid save dir path")
    exit(1)

df = read_file(path)

ratings, sentiments = get_ratings_and_sentiments(df)

# Instantiate a SentimentModality, it makes it convenient to work with sentiment information
md = SentimentModality(data=sentiments)


split_data = RatioSplit(data=ratings, test_size=test_ratio, val_size=validate_ratio,
                        seed=seed_num, verbose=True, sentiment=md)

# configure efm model
r = args.r
r_prime = args.r_prime
# top-k number
k = 15
N = 5
lambda_x = 1
lambda_y = 1
lambda_u = 0.01
lambda_h = 0.01
lambda_v = 0.01
model = cornac.models.EFM(num_explicit_factors=r, num_latent_factors=r_prime,
                          num_most_cared_aspects=k, rating_scale=N,
                          lambda_x=lambda_x, lambda_y=lambda_y,
                          lambda_u=lambda_u, lambda_h=lambda_h,
                          lambda_v=lambda_v, trainable=True, verbose=True,
                          seed=seed_num)
print(f"DataSet {dataset_name}: Now training with r = {r}, r' = {r_prime}")

# performance evaluation metrics: ndcg, recall @ 5, 10, 20, 50
sample_size = [5, 10, 20, 50]
ndcg = [cornac.metrics.NDCG(k=k) for k in sample_size]
recall = [cornac.metrics.Recall(k=k) for k in sample_size]

# perform cross-validation
cornac.Experiment(eval_method=split_data, models=[model], metrics=(ndcg + recall),
                  save_dir=res_path, save_model=False, dataset_name=dataset_name).run()

# save inputs, outputs
elements = ["X", "Y", "A", "U1", "U2", "V", "H1", "H2"]
if save_matrices:
    print(f"Result matrices will be stored in dir {res_path}")
    matrices_file = open(res_path + "matrices.npy", "wb")
    dimension_file = open(res_path + "dimensions.txt", "w")
    dimension_file.write("Inputs:\n")
for element in elements:
    matrix = getattr(model, element)
    if save_matrices:
        np.save(matrices_file, matrix)

    if element == "U1":
        text = "Outputs:\n"
        print(text, end="")
        if save_matrices:
            dimension_file.write(text)

    dim = f"{element}: {matrix.shape}\n"
    print(dim, end="")
    if save_matrices:
        dimension_file.write(dim)

if save_matrices:
    matrices_file.close()
    dimension_file.close()
