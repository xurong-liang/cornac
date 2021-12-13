# perform EFM on a dataset
# requires: 1. reviews.pickle, as per obtained from 7. matched.py (Provide the
# path to the file)
#           2. The directory where the experiment results can be saved

import cornac
import sys
import os

import numpy as np
from cornac.data import SentimentModality
from cornac.eval_methods import RatioSplit
from common_func import read_file, get_ratings_and_sentiments, test_ratio, validate_ratio


seed_num = 1
if len(sys.argv) != 4:
    print("python EFM.py path_to_reviews.pickle path_to_save_evaluation_res "
          "save_resultant_matrices: yes/no")
    exit(1)
path, res_path, save_matrices = sys.argv[1], sys.argv[2], sys.argv[3].lower()

save_matrices = True if save_matrices == "yes" else False

if not os.path.isfile(path):
    print("Invalid input file path")
    exit(1)
if not os.path.isdir(res_path):
    print("Invalid save dir path")
    exit(1)

df, dataset_name = read_file(path)

ratings, sentiments = get_ratings_and_sentiments(df)

# Instantiate a SentimentModality, it makes it convenient to work with sentiment information
md = SentimentModality(data=sentiments)

# # use 5-CV
# cv = CrossValidation(data=ratings, n_folds=5, seed=seed_num, sentiment=md,
#                      verbose=True)


split_data = RatioSplit(data=ratings, test_size=test_ratio, val_size=validate_ratio,
                        seed=seed_num, verbose=True, sentiment=md)

# configure efm model
r = 40
r_prime = 60
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
