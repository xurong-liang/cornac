"""
Functions shared by all EFM-related scripts
"""
import pandas as pd


# .8, .1, .1 proportion
test_ratio, validate_ratio = .1, .1


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
