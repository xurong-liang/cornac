# perform EFM on a dataset
# requires: reviews.pickle, as per obtained from 7. matched.py (Provide the
# path to the file)

import cornac
import sys
import os
import pandas as pd
from cornac.data import SentimentModality
from cornac.eval_methods import CrossValidation, RatioSplit


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
if len(sys.argv) != 2:
    print("Please specify the path to file reviews.pickle")
    exit(1)
path = sys.argv[1]
if not os.path.isfile(path):
    print("Invalid file path")
    exit(1)

df = read_file(path)

ratings, sentiments = get_ratings_and_sentiments(df)

# Instantiate a SentimentModality, it makes it convenient to work with sentiment information
md = SentimentModality(data=sentiments)

# use 5-CV
cv = CrossValidation(data=ratings, n_folds=5, seed=seed_num, sentiment=md,
                     verbose=True)

split_data = RatioSplit(
    data=ratings,
    test_size=0.15,
    exclude_unknowns=True,
    verbose=False,
    sentiment=md,
    seed=seed_num,
)

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

# performance evaluation metrics
ndcg = cornac.metrics.NDCG(k=50)
auc = cornac.metrics.AUC()
recall = cornac.metrics.Recall(k=50)

# perform cross-validation
cornac.Experiment(eval_method=cv, models=[model], metrics=[ndcg, auc, recall]).run()


# do single round of random split to get inputs and outputs matrices
model.verbose = False
cornac.Experiment(eval_method=split_data, models=[model], metrics=[], verbose=False).run()

# output shape of X, Y, A, U1, U2, V, H1, H2
print("\n\n\nFollowing are shapes of model inputs:")
print(f"X: {model.X.shape}")
print(f"Y: {model.Y.shape}")
print(f"A: {model.A.shape}")

print()
print("Following are shapes of model outputs:")
print(f"U1: {model.U1.shape}")
print(f"U2: {model.U2.shape}")
print(f"V: {model.V.shape}")
print(f"H1: {model.H1.shape}")
print(f"H2: {model.H2.shape}")




