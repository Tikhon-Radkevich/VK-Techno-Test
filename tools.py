import numpy as np
from sklearn.metrics import ndcg_score


def calculate_ndcg_per_query(test_data, predictions, k=5, pct_k=0.0):
    test_data_with_preds = test_data.copy()
    test_data_with_preds["preds"] = predictions

    test_data_with_preds = test_data_with_preds.sort_values(["query_id", "rank"], ascending=[True, False])

    ndcg_scores = []
    for query_id, group in test_data_with_preds.groupby("query_id"):
        y_true = group["rank"].tolist()
        y_pred = group["preds"].tolist()
        ndcg_k = min(k, len(y_true))
        if pct_k > 0.0:
            ndcg_k = max(ndcg_k, int(len(y_true) * pct_k))
        ndcg = ndcg_score([y_true], [y_pred], k=ndcg_k)
        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores)


def train_valid_test_split(df, test_size=0.2, valid_size=0.1):
    # remove top 6 longest queries from test and validation sets
    unique_query_ids = df.value_counts("query_id").sort_values(ascending=False)[6:].keys().values

    adjusted_test_size = int(test_size * len(unique_query_ids))
    adjusted_valid_size = int(valid_size * len(unique_query_ids))

    test_query_ids = np.random.choice(unique_query_ids, size=adjusted_test_size, replace=False)
    remaining_query_ids = [q for q in unique_query_ids if q not in test_query_ids]
    valid_query_ids = np.random.choice(remaining_query_ids, size=adjusted_valid_size, replace=False)

    train = df[~df["query_id"].isin(np.concatenate([test_query_ids, valid_query_ids]))]
    test = df[df["query_id"].isin(test_query_ids)]
    valid = df[df["query_id"].isin(valid_query_ids)]

    return train, valid, test
