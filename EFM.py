# perform EFM on a dataset
# requires: 1. reviews.pickle, as per obtained from 7. matched.py (Provide the
# path to the file)
#           2. The directory where the experiment results can be saved

import cornac
import sys
import os

import numpy as np
import pandas as pd
from cornac.data import SentimentModality
from cornac.eval_methods import CrossValidation, RatioSplit, StratifiedSplit


def read_file(file_path: str):
    """
    Read pickle file and add up necessary key-value pairs
    """
    # output from 7.match.py -- mapping of record -> identified sentiment sentences
    df = pd.read_pickle(file_path)

    has_sentence, no_sentence = 0, 0
    for record in df:
        if not record.get("sentence"):
            no_sentence += 1
        else:
            has_sentence += 1
    print(f"no sentence = {no_sentence}; has sentence = {has_sentence}")

    # construct feature-opinion pair mapping (F, S') for each document
    for record in df:
        sentences = record.get("sentence")
        if not sentences:
            continue
        feature_opinion_pair = set()
        feature_mentioned_count = dict()
        feature_mentioned_sentiments = dict()
        for sentence in sentences:
            feature_opinion_pair.add((sentence[0], sentence[-1]))
            if not feature_mentioned_count.get(sentence[0]):
                feature_mentioned_count[sentence[0]] = 1
                feature_mentioned_sentiments[sentence[0]] = sentence[-1]
            else:
                feature_mentioned_count[sentence[0]] += 1
                feature_mentioned_sentiments[sentence[0]] += sentence[-1]

        record["(F, S')"] = feature_opinion_pair
        record["feature_counts"] = feature_mentioned_count
        record["feature_sentiments"] = feature_mentioned_sentiments
    return df


def get_ratings_and_sentiments(df):
    """
    Rating entry: (user, item, rating)
    Sentiment entry: (user, item, [(feature, opinion, str(sentiment))]
    """
    sentiment = []
    rating = []
    for record in df:
        rating.append((record["user"], record["item"], record["rating"]))

        sent = [record["user"], record["item"]]

        triplets = []
        if record.get("sentence"):
            for i in record["sentence"]:
                triplets.append((i[0], i[1], str(i[-1])))
        sent.append(triplets)
        sentiment.append(tuple(sent))
    return rating, sentiment


seed_num = 1
if len(sys.argv) != 3:
    print("Please specify the path to file reviews.pickle and the path to save experiment results")
    exit(1)
path, res_path = sys.argv[1], sys.argv[2]

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

# # use 5-CV
# cv = CrossValidation(data=ratings, n_folds=5, seed=seed_num, sentiment=md,
#                      verbose=True)


# .8, .1, .1 proportion
test_ratio, validate_ratio = .1, .1
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
cornac.Experiment(eval_method=split_data, models=[model], metrics=(ndcg + recall)).run()

# save inputs, outputs
elements = ["X", "Y", "A", "U1", "U2", "V", "H1", "H2"]
matrices_file = open(res_path + "matrices.npy", "wb")
dimension_file = open(res_path + "dimensions.txt", "w")
dimension_file.write("Inputs:\n")
for element in elements:
    matrix = getattr(model, element)
    np.save(matrices_file, matrix)

    if element == "U1":
        text = "Outputs:\n"
        print(text, end="")
        dimension_file.write(text)

    dim = f"{element}: {matrix.shape}\n"
    print(dim, end="")
    dimension_file.write(dim)

matrices_file.close()
dimension_file.close()
