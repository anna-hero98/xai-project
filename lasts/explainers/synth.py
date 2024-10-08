import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.metrics import accuracy_score
import warnings
from tensorflow.keras.utils import to_categorical
from joblib import load, dump
from base import Surrogate, Plotter, SaliencyBasedExplanation
from plots import plot_feature_importance_on_ts


class SynthPatternClassifier(Surrogate):
    def __init__(self, feature_dict, thresholds="auto", balance_default_class=False):
        self.feature_dict = feature_dict
        self.balance_default_class = balance_default_class
        self.thresholds = thresholds
        self.plotter = SynthPatternClassifierPlotter(self)

        self.X_ = None
        self.y_ = None

    def _compute_thresholds(self, bins=100):
        thresholds = dict()
        for key in self.feature_dict.keys():
            thresholds[key] = list()
            for idx, feature in enumerate(self.feature_dict[key].feature_list):
                if isinstance(feature, LocalPattern):
                    X_detrended = np.empty_like(self.X_[:, :, 0])
                    for i, x in enumerate(self.X_):
                        X_detrended[i] = x.ravel() - find_linear_trend(x.ravel())[0]
                    distances = sliding_window_distances(X_detrended, feature.feature)[
                        0
                    ]
                    hist, bin_edges = np.histogram(distances, bins=bins)
                    count = 0
                    lower_idx = 0
                    while count < feature.count:
                        count += hist[lower_idx]
                        lower_idx += 1
                    lower_idx += 1
                    upper_idx = len(hist) - 1
                    count = 0
                    while count < len(self.y_) - feature.count:
                        count += hist[upper_idx]
                        upper_idx -= 1
                    upper_idx -= 1
                    threshold = (bin_edges[lower_idx] + bin_edges[upper_idx]) / 2
                    thresholds[key].append(threshold)
            thresholds[key] = np.array(thresholds[key])
        return thresholds

    def fit(self, X, y):
        assert len(np.unique(y)) == 2  # requires binary labels
        self.X_ = X
        self.y_ = y
        self.thresholds = (
            self._compute_thresholds() if self.thresholds == "auto" else self.thresholds
        )

    def predict(self, X):
        X = X[:, :, 0]
        X_detrended = np.empty_like(X)
        y = np.zeros(shape=len(X))
        for i, x in enumerate(X):
            X_detrended[i] = x - find_linear_trend(x)[0]
        for idx, feature in enumerate(self.feature_dict[1].feature_list):
            if isinstance(feature, LocalPattern):
                y += 1 * pattern_is_contained(
                    X_detrended, feature.feature, self.thresholds[1][idx]
                )
        y = 1 * (y > 0)
        if self.balance_default_class:
            y_0 = np.nonzero(y == 0)  # idxs where y_ == 0
            check = y[y_0].copy()
            # the following checks if there are any sequences of the 0 class in the time series classified as 0
            for idx, feature in enumerate(self.feature_dict[0].feature_list):
                if isinstance(feature, LocalPattern):
                    check += 1 * pattern_is_contained(
                        X_detrended[y_0], feature.feature, self.thresholds[0][idx]
                    )
            # idxs of time series classified as 0 but not having any subsequence of the class 0
            check_0 = np.nonzero(check == 0)
            if len(check_0[0]) != 0:
                X_uncertain = X_detrended[
                    y_0[0][check_0]
                ]  # time series that don't contain any subsequence
                min_distances = np.empty(shape=(len(X_uncertain), 2))
                for key in self.feature_dict.keys():
                    distances_matrix = list()
                    for idx, feature in enumerate(self.feature_dict[key].feature_list):
                        if isinstance(feature, LocalPattern):
                            distances = sliding_window_distances(
                                X_uncertain, feature.feature
                            )[0]
                            distances_matrix.append(distances)
                    min_distances[:, key] = np.min(distances_matrix, axis=0)
                # the class is chosen by looking at the closest (not contained) subsequence
                y_uncertain = np.argmin(min_distances, axis=1)
                y[y_0[0][check_0]] = y_uncertain
        return y

    def predict_proba(self, X):
        y = self.predict(X)
        return to_categorical(y, num_classes=2)

    def explain(self, x):
        x = x.ravel()
        x_label = self.predict(x.reshape(1, -1, 1))[0]
        trend = find_linear_trend(x.ravel())[0]
        x_detrend = x.ravel() - trend

        min_dist_check = np.inf
        min_feature = None
        min_idx_check = None
        check = False
        global_dummy_ts = np.repeat(np.nan, len(x))
        for i, feature in enumerate(self.feature_dict[x_label].feature_list):
            if isinstance(feature, LocalPattern):
                min_dist, min_idx = sliding_window_distance(x_detrend, feature.feature)
                if min_dist < min_dist_check:
                    min_dist_check = min_dist
                    min_idx_check = min_idx
                    min_feature = feature.feature
                if min_dist <= self.thresholds[x_label][i]:
                    check = check or True
                    global_dummy_ts[
                        min_idx : min_idx + len(feature.feature)
                    ] = feature.feature

        if not check and self.balance_default_class:
            global_dummy_ts[
                min_idx_check : min_idx_check + len(min_feature)
            ] = min_feature
        alignment_ts = np.nan_to_num((global_dummy_ts * 0) + 1).reshape(1, -1, 1)
        explanation = SaliencyBasedExplanation(alignment_ts)
        return explanation

    def plot(self, kind, **kwargs):
        self.plotter.plot(kind, **kwargs)

    def evaluate(self, metric, **kwargs):
        pass


class SynthPatternClassifierPlotter(Plotter):
    def __init__(self, synth_clf: SynthPatternClassifier):
        self.synth_clf = synth_clf

    def plot(self, kind, **kwargs):
        pass

    def plot_thresholds_histograms(self, bins=100):
        for key in feature_dict.keys():
            for idx, feature in enumerate(feature_dict[key].feature_list):
                if isinstance(feature, LocalPattern):
                    distances = np.empty(shape=len(self.synth_clf.X_))
                    for i, x in enumerate(self.synth_clf.X_):
                        x_detrend = x.ravel() - find_linear_trend(x.ravel())[0]
                        distances[i] = sliding_window_distance(
                            x_detrend, feature.feature
                        )[0]
                    plt.title(str(key) + " - " + str(idx))
                    plt.hist(distances, bins=bins)
                    plt.axvline(x=self.synth_clf.thresholds[key][idx], c="red")
                    plt.show()
        print(self.synth_clf.thresholds)
        return self

    def plot_saliency(self, x, **kwargs):
        plot_feature_importance_on_ts(
            x.ravel(), self.synth_clf.explain(x).saliency.ravel(), **kwargs
        )

    def plot_pattern(self, x, figsize=(10, 4), dpi=72, fontsize=18):
        x = x.ravel()
        x_label = self.synth_clf.predict(x.reshape(1, -1, 1))[0]
        trend = find_linear_trend(x.ravel())[0]
        x_detrend = x.ravel() - trend
        # colors = ["red", "green"]
        color = "#2ca02c"
        plt.figure(figsize=figsize, dpi=dpi)
        plt.title(
            "Ground Truth - " + r"$b(x)$" + " = " + str(x_label), fontsize=fontsize
        )
        plt.plot(x, c="royalblue", alpha=0.2, lw=3)

        min_dist_check = np.inf
        min_feature = None
        min_idx_check = None
        check = False
        for i, feature in enumerate(self.synth_clf.feature_dict[x_label].feature_list):
            if isinstance(feature, LocalPattern):
                min_dist, min_idx = sliding_window_distance(x_detrend, feature.feature)
                if min_dist < min_dist_check:
                    min_dist_check = min_dist
                    min_idx_check = min_idx
                    min_feature = feature.feature
                if min_dist <= self.synth_clf.thresholds[x_label][i]:
                    check = check or True
                    dummy_ts = np.repeat(np.nan, len(x))
                    dummy_ts[min_idx : min_idx + len(feature.feature)] = feature.feature
                    dummy_ts += trend
                    plt.plot(dummy_ts, c=color, alpha=0.5, lw=5)
        if not check and self.synth_clf.balance_default_class:
            dummy_ts = np.repeat(np.nan, len(x))
            dummy_ts[min_idx_check : min_idx_check + len(min_feature)] = min_feature
            dummy_ts += trend
            plt.plot(dummy_ts, c="blue", alpha=0.5, lw=5)
        plt.ylabel("value", fontsize=fontsize)
        plt.xlabel("time-steps", fontsize=fontsize)
        plt.tick_params(axis="both", which="major", labelsize=fontsize)
        plt.show()
        return self


# class PatternClassifier(object):
#     def __init__(self, feature_dict, X, y, thresholds="auto", balance_default_class=False):
#         super(PatternClassifier, self).__init__()
#         self.feature_dict = feature_dict
#         self.X = X
#         assert len(np.unique(y)) == 2  # requires binary labels
#         self.y = y
#         self.thresholds = self._compute_thresholds() if thresholds == "auto" else thresholds
#         self.balance_default_class = balance_default_class
#
#     def _compute_thresholds(self, bins=100):
#         thresholds = dict()
#         for key in feature_dict.keys():
#             thresholds[key] = list()
#             for idx, feature in enumerate(feature_dict[key].feature_list):
#                 if isinstance(feature, LocalPattern):
#                     X_detrended = np.empty_like(self.X[:, :, 0])
#                     for i, x in enumerate(self.X):
#                         X_detrended[i] = x.ravel() - find_linear_trend(x.ravel())[0]
#                     distances = sliding_window_distances(X_detrended, feature.feature)[0]
#                     hist, bin_edges = np.histogram(distances, bins=bins)
#                     count = 0
#                     lower_idx = 0
#                     while count < feature.count:
#                         count += hist[lower_idx]
#                         lower_idx += 1
#                     lower_idx += 1
#                     upper_idx = len(hist) - 1
#                     count = 0
#                     while count < len(self.y) - feature.count:
#                         count += hist[upper_idx]
#                         upper_idx -= 1
#                     upper_idx -= 1
#                     threshold = (bin_edges[lower_idx] + bin_edges[upper_idx]) / 2
#                     thresholds[key].append(threshold)
#             thresholds[key] = np.array(thresholds[key])
#         return thresholds
#
#     def _plot_thresholds_histograms(self, bins=100):
#         for key in feature_dict.keys():
#             for idx, feature in enumerate(feature_dict[key].feature_list):
#                 if isinstance(feature, LocalPattern):
#                     distances = np.empty(shape=len(self.X))
#                     for i, x in enumerate(self.X):
#                         x_detrend = x.ravel() - find_linear_trend(x.ravel())[0]
#                         distances[i] = (sliding_window_distance(x_detrend, feature.feature)[0])
#                     plt.title(str(key) + " - " + str(idx))
#                     plt.hist(distances, bins=bins)
#                     plt.axvline(x=self.thresholds[key][idx], c="red")
#                     plt.show()
#         print(self.thresholds)
#         return self
#
#     def predict(self, X):
#         X = X[:, :, 0]
#         X_detrended = np.empty_like(X)
#         y = np.zeros(shape=len(X))
#         for i, x in enumerate(X):
#             X_detrended[i] = x - find_linear_trend(x)[0]
#         for idx, feature in enumerate(self.feature_dict[1].feature_list):
#             if isinstance(feature, LocalPattern):
#                 y += 1 * pattern_is_contained(X_detrended, feature.feature, self.thresholds[1][idx])
#         y = 1 * (y > 0)
#
#         if self.balance_default_class:
#             y_0 = np.nonzero(y == 0)  # idxs where y_ == 0
#             check = y[y_0].copy()
#             # the following checks if there are any sequences of the 0 class in the time series classified as 0
#             for idx, feature in enumerate(self.feature_dict[0].feature_list):
#                 if isinstance(feature, LocalPattern):
#                     check += 1 * pattern_is_contained(X_detrended[y_0], feature.feature, self.thresholds[0][idx])
#             # idxs of time series classified as 0 but not having any subsequence of the class 0
#             check_0 = np.nonzero(check == 0)
#             if len(check_0[0]) != 0:
#                 X_uncertain = X_detrended[y_0[0][check_0]]  # time series that don't contain any subsequence
#                 min_distances = np.empty(shape=(len(X_uncertain), 2))
#                 for key in self.feature_dict.keys():
#                     distances_matrix = list()
#                     for idx, feature in enumerate(self.feature_dict[key].feature_list):
#                         if isinstance(feature, LocalPattern):
#                             distances = sliding_window_distances(X_uncertain, feature.feature)[0]
#                             distances_matrix.append(distances)
#                     min_distances[:, key] = np.min(distances_matrix, axis=0)
#                 # the class is chosen by looking at the closest (not contained) subsequence
#                 y_uncertain = np.argmin(min_distances, axis=1)
#                 y[y_0[0][check_0]] = y_uncertain
#         return y
#
#     def predict_proba(self, X):
#         y = self.predict(X)
#         return to_categorical(y, num_classes=2)
#
#     def score(self, X, y):
#         return accuracy_score(y, self.predict(X))
#
#     def explain(self, x, figsize=(10, 4), dpi=72, fontsize=18):
#         x = x.ravel()
#         x_label = self.predict(x.reshape(1, -1, 1))[0]
#         trend = find_linear_trend(x.ravel())[0]
#         x_detrend = x.ravel() - trend
#         #colors = ["red", "green"]
#         color = "#2ca02c"
#         plt.figure(figsize=figsize, dpi=dpi)
#         plt.title("Ground Truth - " + r"$b(x)$" + " = " + str(x_label), fontsize=fontsize)
#         plt.plot(x, c="royalblue", alpha=0.2, lw=3)
#
#         min_dist_check = np.inf
#         min_feature = None
#         min_idx_check = None
#         check = False
#         for i, feature in enumerate(self.feature_dict[x_label].feature_list):
#             if isinstance(feature, LocalPattern):
#                 min_dist, min_idx = sliding_window_distance(x_detrend, feature.feature)
#                 if min_dist < min_dist_check:
#                     min_dist_check = min_dist
#                     min_idx_check = min_idx
#                     min_feature = feature.feature
#                 if min_dist <= self.thresholds[x_label][i]:
#                     check = check or True
#                     dummy_ts = np.repeat(np.nan, len(x))
#                     dummy_ts[min_idx:min_idx + len(feature.feature)] = feature.feature
#                     dummy_ts += trend
#                     plt.plot(dummy_ts, c=color, alpha=0.5, lw=5)
#         if not check and self.balance_default_class:
#             dummy_ts = np.repeat(np.nan, len(x))
#             dummy_ts[min_idx_check:min_idx_check + len(min_feature)] = min_feature
#             dummy_ts += trend
#             plt.plot(dummy_ts, c="blue", alpha=0.5, lw=5)
#         plt.ylabel("value", fontsize=fontsize)
#         plt.xlabel("time-steps", fontsize=fontsize)
#         plt.tick_params(axis='both', which='major', labelsize=fontsize)
#         plt.show()
#         return self
#
#     def predict_explanation(self, x):
#         x = x.ravel()
#         x_label = self.predict(x.reshape(1, -1, 1))[0]
#         trend = find_linear_trend(x.ravel())[0]
#         x_detrend = x.ravel() - trend
#
#         min_dist_check = np.inf
#         min_feature = None
#         min_idx_check = None
#         check = False
#         global_dummy_ts = np.repeat(np.nan, len(x))
#         for i, feature in enumerate(self.feature_dict[x_label].feature_list):
#             if isinstance(feature, LocalPattern):
#                 min_dist, min_idx = sliding_window_distance(x_detrend, feature.feature)
#                 if min_dist < min_dist_check:
#                     min_dist_check = min_dist
#                     min_idx_check = min_idx
#                     min_feature = feature.feature
#                 if min_dist <= self.thresholds[x_label][i]:
#                     check = check or True
#                     global_dummy_ts[min_idx:min_idx + len(feature.feature)] = feature.feature
#
#         if not check and self.balance_default_class:
#             global_dummy_ts[min_idx_check:min_idx_check + len(min_feature)] = min_feature
#         alignment_ts = np.nan_to_num((global_dummy_ts * 0) + 1)
#         return alignment_ts


def random_pattern_modification(pat, random_state=None, min_dim=15):
    if random_state is not None:
        np.random.seed(random_state)
    for j in range(2):
        for i, feature in enumerate(pat.feature_dict[j].feature_list):
            if not isinstance(feature, LocalPattern):
                continue
            new_pattern_length = np.random.randint(min_dim, feature.length)
            start_idx = np.random.randint(0, feature.length - new_pattern_length)
            new_pattern = feature.feature[start_idx : start_idx + new_pattern_length]
            pat.feature_dict[j].feature_list[i].feature = new_pattern
            pat.feature_dict[j].feature_list[i].length = new_pattern_length
    pat._compute_thresholds()
    return


def create_random_pat(base_pat_path, n=5, min_dim=15):
    for i in range(n):
        pat = load(base_pat_path)
        random_pattern_modification(pat, random_state=i, min_dim=min_dim)
        print(i, pat.score(pat.X_, pat.y_))
        dump(pat, base_pat_path[:-7] + str(i) + ".joblib")
    return


def pattern_is_contained(tss, s, threshold):
    min_distances, _ = sliding_window_distances(tss, s)
    return min_distances <= threshold


def sliding_window_distance(ts, s):
    m = len(s)
    dist = np.square(s[0] - ts[:-m])
    for i in range(1, m):
        dist += np.square(s[i] - ts[i : -m + i])
    dist = np.sqrt(dist) / m
    return np.min(dist), np.argmin(dist)


def sliding_window_distances(tss, s):
    m = len(s)
    dist = np.square(s[0] - tss[:, :-m])
    for i in range(1, m):
        dist += np.square(s[i] - tss[:, i : -m + i])
    dist = np.sqrt(dist) / m
    return np.min(dist, axis=1), np.argmin(dist, axis=1)


def shuffle_arrays(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def find_linear_trend(ts):
    p = np.polyfit(x=np.arange(len(ts)), y=ts, deg=1)
    trend = (np.arange(len(ts)) * p[0]) + p[1]
    return trend, p[0], p[1]


def generate_random_walk(
    length=20, kind="continuous", step_set=(-1, 0, 1), mu=0, sigma=1, random_state=None
):
    if random_state is not None:
        np.random.seed(random_state)
    if kind == "continuous":
        steps = np.random.normal(loc=mu, scale=sigma, size=(length,))
    elif kind == "discrete":
        steps = np.random.choice(a=step_set, size=(length,))
    else:
        raise Exception("kind is not valid.")
    path = steps.cumsum(axis=0)
    return path


def generate_sine_wave(length=20, min_frequency=1, sample_rate=100, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    x = np.arange(length)
    frequency = np.random.uniform(min_frequency, sample_rate / 10)
    pattern = np.sin(2 * np.pi * frequency * x / sample_rate)
    return pattern


def generate_pattern(
    kind="random_walk",
    length=20,
    normalize=True,
    custom_pattern=None,
    generator_kwargs=dict(),
    **kwargs
):
    if kind == "custom":
        pattern = custom_pattern
    elif kind == "random_walk":
        pattern = generate_random_walk(length=length, **generator_kwargs)
    elif kind == "sine_wave":
        pattern = generate_sine_wave(length=length, **generator_kwargs)
    else:
        raise Exception("kind is not valid.")
    if normalize:
        pattern = zscore(pattern)
    return pattern


def constant_feature(length=200, min=0, max=1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    ts = np.repeat(np.random.uniform(min, max), length)
    return ts


def linear_trend(length=200, min=0.005, max=0.01, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    ts = np.cumsum(np.repeat(np.random.uniform(min, max), length))
    return ts


def linear_polynomial(
    length=200, p0_min=0.1, p0_max=1, p1_min=0.001, p1_max=0.01, random_state=None
):
    if random_state is not None:
        np.random.seed(random_state)
    p0 = np.random.uniform(p0_min, p0_max)
    p1 = np.random.uniform(p1_min, p1_max)
    ts = (np.arange(length) * p1) + p0
    return ts


def white_noise(length=200, mu=0, sigma=1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    ts = np.random.normal(loc=mu, scale=sigma, size=(length,))
    return ts


def generate_global_feature(
    kind="constant", length=200, generator_kwargs=dict(), **kwargs
):
    if kind == "constant":
        ts = constant_feature(length, **generator_kwargs)
    elif kind == "white_noise":
        ts = white_noise(length, **generator_kwargs)
    elif kind == "linear_trend":
        ts = linear_trend(length, **generator_kwargs)
    elif kind == "linear_polynomial":
        ts = linear_polynomial(length, **generator_kwargs)
    else:
        raise Exception("kind is not valid.")
    return ts


class DatasetGenerator(object):
    def __init__(self, n, length, feature_dict, reset_feature_counts=True):
        self.n = n
        self.feature_dict = feature_dict
        self.length = length
        if reset_feature_counts:
            for key in feature_dict:
                for feature in feature_dict[key].feature_list:
                    feature.count = 0

    def generate(self, shuffle=True, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        X = list()
        y = list()
        for label in self.feature_dict.keys():
            for i in range(self.n):
                X.append(self.feature_dict[label].generate())
                self.feature_dict[label].reset()
                y.append(label)
        X = np.array(X)
        y = np.array(y)
        if shuffle:
            X, y = shuffle_arrays(X, y)
        return X[:, :, np.newaxis], y

    def _plot_patterns_by_label(self):
        colors = ["red", "green"]
        for key in feature_dict:
            for feature in feature_dict[key].feature_list:
                if isinstance(feature, LocalPattern):
                    plt.plot(feature.feature, c=colors[key])
        plt.show()


class TimeSeriesGenerator(object):
    def __init__(self, length, feature_list):
        self.length = length
        self.feature_list = feature_list
        self.free_idxs = np.arange(0, length)

    def generate(self, local_feature_skip_probability=0.5, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        ts = np.zeros(self.length)
        local_features_count = 0
        for feature in self.feature_list:
            if isinstance(feature, LocalPattern):
                local_features_count += 1
        for feature in self.feature_list:
            if isinstance(feature, LocalPattern):
                if local_features_count >= 2:
                    if np.random.random() < local_feature_skip_probability:
                        local_features_count -= 1
                        continue
                if not feature.avoid_pattern_overlap:
                    start_idx = np.random.choice(np.arange(len(ts) - feature.length))
                else:
                    possible_starting_idxs = list()
                    consecutive_idxs_groups = np.split(
                        self.free_idxs, np.where(np.diff(self.free_idxs) != 1)[0] + 1
                    )
                    check = False
                    for group in consecutive_idxs_groups:
                        check = check or (len(group) >= feature.length)
                        if len(group) >= feature.length:
                            if len(group) == feature.length:
                                possible_starting_idxs.append(group[0])
                            else:
                                possible_starting_idxs.extend(
                                    np.arange(
                                        group[0],
                                        group[0] + (len(group) - feature.length),
                                    )
                                )
                    if not check:
                        raise Exception("Not enough space for a new pattern.")
                    start_idx = np.random.choice(possible_starting_idxs)
                    end_idx = start_idx + feature.length
                    if end_idx == self.length:
                        end_idx -= 1
                    to_delete = np.arange(start_idx, end_idx)
                    for value in to_delete:
                        self.free_idxs = np.delete(
                            self.free_idxs, np.argwhere(self.free_idxs == value)
                        )
                ts = feature.apply(ts, start_idx)
                feature.count += 1
            else:
                ts = feature.apply(ts)
        return ts

    def reset(self):
        self.free_idxs = np.arange(0, self.length)
        return self

    def plot_patterns(self):
        for feature in self.feature_list:
            if isinstance(feature, LocalPattern):
                plt.plot(feature.feature)
        plt.show()
        return self


class FeatureGenerator(object):
    def __init__(self, kind):
        self.kind = kind
        self.feature = None
        self.count = 0
        pass

    def build(self):
        pass

    def apply(self, ts):
        pass


class LocalPattern(FeatureGenerator):
    def __init__(self, kind, length, avoid_pattern_overlap=True, **kwargs):
        super(LocalPattern, self).__init__(kind)
        self.length = length
        self.avoid_pattern_overlap = avoid_pattern_overlap
        self.build(**kwargs)

    def build(self, **kwargs):
        pattern = generate_pattern(self.kind, self.length, **kwargs)
        self.feature = pattern
        return self

    def apply(self, ts, start_idx):
        ts_new = ts.copy()
        end_idx = start_idx + self.length
        ts_new[start_idx:end_idx] = self.feature
        return ts_new


class GlobalFeature(FeatureGenerator):
    def __init__(self, kind, length, **kwargs):
        super(GlobalFeature, self).__init__(kind)
        self.length = length
        self.build(**kwargs)

    def build(self, **kwargs):
        self.feature = generate_global_feature(self.kind, self.length, **kwargs)
        return self

    def apply(self, ts):
        ts_new = ts.copy()
        ts_new += self.feature
        return ts_new


if __name__ == "__main__":
    np.random.seed(1)
    length = 96
    feature_list1 = [
        LocalPattern(kind="sine_wave", length=40),
        GlobalFeature(
            kind="white_noise", length=length, generator_kwargs={"sigma": 0.1}
        ),
    ]

    feature_list2 = [
        LocalPattern(kind="sine_wave", length=40),
        GlobalFeature(
            kind="white_noise", length=length, generator_kwargs={"sigma": 0.1}
        ),
    ]

    feature_dict = {
        0: TimeSeriesGenerator(length=length, feature_list=feature_list1),
        1: TimeSeriesGenerator(length=length, feature_list=feature_list2),
    }

    datagen = DatasetGenerator(n=300, length=length, feature_dict=feature_dict)

    X, y = datagen.generate(random_state=0)
    i = 0
    pat = SynthPatternClassifier(
        datagen.feature_dict, thresholds="auto", balance_default_class=True
    )
    pat.fit(X, y)
    print(pat.score(X, y))
    pat.explain(X[i])
    pat.plotter.plot_pattern(X[i])
    pat.plotter.plot_saliency(X[i])
    dump(pat, "synth_pat0.joblib")

    # random_pattern_modification(pat, random_state=5)
    # print(pat.score(X, y))
    # pat.explain(X[i])
    # pat.plotter.plot_pattern(X[i])
    # pat.plotter.plot_saliency(X[i])
    #
    #
    # create_random_pat("./trained_models/synth04/synth04_pat.joblib", min_dim=10)
    #
    # pat0 = load("./trained_models/synth04/synth04_pat0.joblib")
    # pat0.explain(X_[i])
