import numpy as np
from collections import Counter
import itertools



def find_best_split(feature_vector, target_vector):
    feature_vector = np.array(feature_vector)
    target_vector = np.array(target_vector)

    sort_index = np.argsort(feature_vector)
    thresholds = (feature_vector[sort_index][:(len(feature_vector) - 1)] + feature_vector[sort_index][1:]) / 2
    size_R_l = np.arange(len(thresholds)) + 1
    size_R_r = size_R_l[::-1]
    cumsum_l = np.cumsum(target_vector[sort_index])[:-1]

    Hs_l = 1 - (cumsum_l / size_R_l) ** 2 - ((size_R_l - cumsum_l) / size_R_l) ** 2
    tv_sum = np.sum(target_vector)
    rsum = tv_sum - cumsum_l
    Hs_r = 1 - (rsum / size_R_r) ** 2 - ((size_R_r - rsum) / size_R_r) ** 2
    tmp = feature_vector[sort_index]
    _, count = np.unique(tmp, return_counts=True)
    count = (np.cumsum(count) - 1)[:-1]
    thresholds = thresholds[count]
    ginis = -size_R_l / len(feature_vector) * Hs_l - size_R_r / len(feature_vector) * Hs_r
    ginis = ginis[count]
    best_th = thresholds[np.argmin(np.abs(ginis))]
    best_g = ginis[np.argmin(np.abs(ginis))]
    return thresholds, ginis, best_th, best_g


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])


    def _predict_node(self, x, node):
        if node["type"] != "nonterminal":
            return node["class"]
        else:
            if self._feature_types[node["feature_split"]] == "categorical":
                if x[node["feature_split"]] not in node["categories_split"]:
                    return self._predict_node(x, node["right_child"])
                else:
                    return self._predict_node(x, node["left_child"])

            if self._feature_types[node["feature_split"]] == "real":
                if x[node["feature_split"]] >= node["threshold"]:
                    return self._predict_node(x, node["right_child"])
                else:
                    return self._predict_node(x, node["left_child"])


    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)