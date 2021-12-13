"""
This script is used to tune the hyperparmeters of EFM model. In precise, the
number of explicit (r) and implicit (r') factors are to be tuned

Usage: python EFM_hyperparam_tuning.py path_to_reviews.pickle path_to_save_result (if applicable) verbose: yes/no dataset_name
"""

import sys
import cornac
import os
from common_func import read_file, get_ratings_and_sentiments, test_ratio, validate_ratio

seed_num = 1

arg_num = len(sys.argv)
if arg_num != 5:
    print("python EFM_hyperparam_tuning.py path_to_reviews.pickle path_to_save_result; write None"
          " when N/A verbose: yes/no dataset_name")
    exit(1)

if sys.argv[1].title() == "None":
    # no dataset supplied, use electronics dataset by default
    path = "/media/bigdata/uqxlian4/electronics_processing/English-jar/lei/output/reviews.pickle"
else:
    path = sys.argv[1]

if not os.path.isfile(path):
    print("Invalid input file path")
    exit(1)

if sys.argv[2].title() == "None":
    write_output = False
    res_path = None
else:
    res_path = sys.argv[2]

    if not os.path.isdir(res_path):
        print("Invalid output dir path")
        exit(1)
    write_output = True

verbose = True if sys.argv[3].lower() == "yes" else False

dataset_name = sys.argv[4]

df = read_file(path)

ratings, sentiments = get_ratings_and_sentiments(df)

# Instantiate a SentimentModality, it makes it convenient to work with sentiment information
md = cornac.data.SentimentModality(data=sentiments)

split_data = cornac.eval_methods.RatioSplit(data=ratings, test_size=test_ratio, val_size=validate_ratio,
                                            seed=seed_num, verbose=True, sentiment=md)

# hyperparams that are fixed
# top-k number
k = 15
N = 5
lambda_x = 1
lambda_y = 1
lambda_u = 0.01
lambda_h = 0.01
lambda_v = 0.01

# performance evaluation metrics: ndcg, recall @ 5, 10, 20, 50
sample_size = [5, 10, 20, 50]
ndcg = [cornac.metrics.NDCG(k=k) for k in sample_size]
recall = [cornac.metrics.Recall(k=k) for k in sample_size]
metrics = ndcg + recall

# all settings of r's and r_prime's
rs = [10 * _ for _ in [1, 4, 8, 16]]
r_primes = [10 * _ for _ in [1, 4, 8, 16]]

if write_output:
    fp = open(os.path.join(res_path, f"{dataset_name}_hyperparam_tuning_res.log"), "w")

settings = []

metrics_settings = [f"{name}@{sz}" for sz in sample_size for name in ["NDCG", "Recall"]]
for combination in metrics_settings:
    locals()[combination] = []

print(f"dataset {dataset_name}, now start hyperparam tuning:")
for r in rs:
    for r_prime in r_primes:
        current_setting = f"r = {r}; r' = {r_prime}"
        settings.append(current_setting)
        current_setting += "\n"
        print(current_setting, end="") if verbose else None
        if write_output:
            fp.write(current_setting)
        model = cornac.models.EFM(num_explicit_factors=r, num_latent_factors=r_prime,
                                  num_most_cared_aspects=k, rating_scale=N,
                                  lambda_x=lambda_x, lambda_y=lambda_y,
                                  lambda_u=lambda_u, lambda_h=lambda_h,
                                  lambda_v=lambda_v, trainable=True, verbose=False,
                                  seed=seed_num)
        experiment = current_output = cornac.Experiment(
            eval_method=split_data, models=[model], metrics=metrics,
            save_dir=res_path, save_model=write_output, dataset_name=dataset_name,
            hyper_param_tuning=True, verbose=False)
        experiment.run()

        scores = experiment.result[0].metric_avg_results
        for combination in metrics_settings:
            locals()[combination].append(scores[combination])

        print(current_output, end="") if verbose else None
        if write_output:
            fp.write(current_output)
            fp.write("\n")
if write_output:
    fp.close()

print("\nDetail results:")
for i in range(len(settings)):
    text = f"setting: {settings[i]}, "
    for metric in metrics_settings:
        text += f"{metric} = {locals()[metric][i]}, "
    print(text)
print("\n")

# now print out the best setting for each of the metric settings
for i in metrics_settings:
    metric_scores = locals()[i]
    highest_idx = metric_scores.index(max(metric_scores))
    param_setting = settings[highest_idx]
    metric_score = metric_scores[highest_idx]

    print(f"best setting for evaluation metric {i} = {param_setting}; with "
          f"value = {metric_score:.8f}")
