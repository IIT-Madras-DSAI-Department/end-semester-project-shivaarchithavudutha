import numpy as np
from scipy.linalg import eigh
from scipy.stats import mode
import time

def confusion_matrix(y_true, y_pred, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Filling the matrix
    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred[i]
        matrix[true_label, pred_label] += 1

    return matrix

def macro_f1_score(y_true, y_pred, num_classes=10):
    matrix = confusion_matrix(y_true, y_pred, num_classes)

    f1_scores = []

    for i in range(num_classes):
        tp = matrix[i, i]
        fp = np.sum(matrix[:, i]) - tp
        fn = np.sum(matrix[i, :]) - tp

        # Precision = TP / (TP + FP)
        if (tp + fp) == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        if (tp + fn) == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        f1_scores.append(f1)

    # Macro F1 is the average of all class F1 scores
    return np.mean(f1_scores)

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        n_samples = X.shape[0]
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
        eigenvalues, eigenvectors = eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        self.components = sorted_eigenvectors[:, :self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        X_pca = X_centered @ self.components

        return X_pca

def one_hot_encode(y, num_classes):

    y_one_hot = np.zeros((y.shape[0], num_classes))
    y_one_hot[np.arange(y.shape[0]), y] = 1

    return y_one_hot

def softmax(z):
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    probabilities = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    return probabilities

class SoftmaxRegression:
    def __init__(self, learning_rate, epochs, reg_strength):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_strength = reg_strength
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))
        y_one_hot = one_hot_encode(y, n_classes)

        for i in range(self.epochs):
            scores = X @ self.weights + self.bias
            probabilities = softmax(scores)

            correct_logprobs = -np.log(probabilities[np.arange(n_samples), y] + 1e-9)
            data_loss = np.sum(correct_logprobs) / n_samples

            reg_loss = 0.5 * self.reg_strength * np.sum(self.weights * self.weights)

            total_loss = data_loss + reg_loss

            if (i % 100 == 0):
                print(f"Epoch {i}, Loss: {total_loss:.4f}")

            dscores = probabilities - y_one_hot
            dscores /= n_samples

            dweights = X.T @ dscores
            dbias = np.sum(dscores, axis=0, keepdims=True)

            dweights += self.reg_strength * self.weights

            self.weights -= self.learning_rate * dweights
            self.bias -= self.learning_rate * dbias

    def predict_proba(self, X):
        scores = X @ self.weights + self.bias
        return softmax(scores)

    def predict(self, X):
        probabilities = self.predict_proba(X)

        return np.argmax(probabilities, axis=1)


class LinearRegression:

    def __init__(self, learning_rate, epochs, reg_strength):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_strength = reg_strength
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y = y.reshape(-1)

        # Gradient Descent
        for _ in range(self.epochs):
            # Predict
            y_pred = X @ self.weights + self.bias

            # Calculate error
            error = y_pred - y

            # Calculate gradients
            dw = (X.T @ error) / n_samples
            db = np.sum(error) / n_samples

            # Add L2 regularization gradient
            dw += self.reg_strength * self.weights

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Return the raw scores
        return X @ self.weights + self.bias

class OvRLinearClassifier:
    def __init__(self, learning_rate, epochs, reg_strength, num_classes=10):
        self.num_classes = num_classes
        self.models = []
        # Store hyperparameters for the base model
        self.lr = learning_rate
        self.epochs = epochs
        self.reg = reg_strength

    def fit(self, X, y):
        print("Training OvR Linear Classifier...")
        self.models = []

        for i in range(self.num_classes):
            print(f"  Training model for class {i}...")
            y_binary = (y == i).astype(int)

            model = LinearRegression(learning_rate=self.lr,
                                     epochs=self.epochs,
                                     reg_strength=self.reg)
            model.fit(X, y_binary)
            self.models.append(model)
        print("OvR training complete.")

    def predict_proba(self, X):
        scores = np.zeros((X.shape[0], self.num_classes))
        for i, model in enumerate(self.models):
            scores[:, i] = model.predict(X)

        return softmax(scores)

    def predict(self, X):
        scores = np.zeros((X.shape[0], self.num_classes))
        for i, model in enumerate(self.models):
            scores[:, i] = model.predict(X)
        return np.argmax(scores, axis=1)


class KMeansClassifier:
    def __init__(self, k=10, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.centroid_labels = None

    def _find_closest_centroids(self, X):

        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            distances[:, i] = np.sum((X - self.centroids[i])**2, axis=1)

        # Return the index of the closest centroid for each sample
        return np.argmin(distances, axis=1)

    def fit(self, X, y):
        print("Training K-Means Classifier...")
        n_samples, n_features = X.shape

        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            cluster_assignments = self._find_closest_centroids(X)

            new_centroids = np.zeros((self.k, n_features))
            for j in range(self.k):
                points_in_cluster = X[cluster_assignments == j]
                if len(points_in_cluster) > 0:
                    new_centroids[j] = np.mean(points_in_cluster, axis=0)
                else:
                    new_centroids[j] = X[np.random.choice(n_samples)]

            if np.allclose(self.centroids, new_centroids):
                print(f"Converged at iteration {i}.")
                break

            self.centroids = new_centroids

        print("K-Means clustering complete. Assigning centroid labels...")

        self.centroid_labels = np.zeros(self.k, dtype=int)
        for i in range(self.k):
            labels_in_cluster = y[cluster_assignments == i]

            if len(labels_in_cluster) > 0:
                self.centroid_labels[i] = mode(labels_in_cluster, keepdims=True)[0][0]
            else:
                self.centroid_labels[i] = np.random.choice(10)

        print("K-Means training complete.")

    def predict(self, X):
        cluster_assignments = self._find_closest_centroids(X)

        return self.centroid_labels[cluster_assignments]

    def predict_proba(self, X):
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            distances[:, i] = np.sum((X - self.centroids[i])**2, axis=1)

        closeness = 1.0 / (distances + 1e-6)

        pseudo_proba = closeness / np.sum(closeness, axis=1, keepdims=True)

        proba_output = np.zeros((X.shape[0], 10))

        for i in range(self.k):
            centroid_label = self.centroid_labels[i]
            proba_output[:, centroid_label] += pseudo_proba[:, i]

        row_sums = np.sum(proba_output, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1

        return proba_output / row_sums

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, max_thresholds=25):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.max_thresholds = max_thresholds
        self.root = None

    def _gini_impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities**2)
        return gini

    def _find_best_split(self, X, y, feat_indices):
        best_gain = -1
        best_feature = None
        best_threshold = None
        current_gini = self._gini_impurity(y)

        for feature_index in feat_indices:

            all_thresholds = np.unique(X[:, feature_index])

            if len(all_thresholds) > self.max_thresholds:
                thresholds = np.random.choice(all_thresholds, self.max_thresholds, replace=False)
            else:
                thresholds = all_thresholds

            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                gini_left = self._gini_impurity(y[left_indices])
                gini_right = self._gini_impurity(y[right_indices])

                weight_left = len(left_indices) / len(y)
                weight_right = len(right_indices) / len(y)

                weighted_gini = (weight_left * gini_left) + (weight_right * gini_right)
                information_gain = current_gini - weighted_gini

                if information_gain > best_gain:
                    best_gain = information_gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_labels == 1):
            leaf_value = mode(y, keepdims=True)[0][0]
            return Node(value=leaf_value)

        feat_indices = np.random.choice(n_feats, self.n_features, replace=False) \
                       if self.n_features else np.arange(n_feats)

        feature, threshold = self._find_best_split(X, y, feat_indices)

        if feature is None:
            leaf_value = mode(y, keepdims=True)[0][0]
            return Node(value=leaf_value)

        left_indices = np.where(X[:, feature] <= threshold)[0]
        right_indices = np.where(X[:, feature] > threshold)[0]

        X_left, y_left = X[left_indices, :], y[left_indices]
        X_right, y_right = X[right_indices, :], y[right_indices]

        left_child = self._grow_tree(X_left, y_left, depth + 1)
        right_child = self._grow_tree(X_right, y_right, depth + 1)

        return Node(feature, threshold, left_child, right_child)

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None, max_thresholds=25):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.max_thresholds = max_thresholds # New
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        print("Training Random Forest...")
        self.trees = []

        if self.n_features is None:
            self.n_features = int(np.sqrt(X.shape[1]))

        for i in range(self.n_trees):
            print(f"  Training tree {i+1}/{self.n_trees}...")

            tree = DecisionTree(min_samples_split=self.min_samples_split,
                                max_depth=self.max_depth,
                                n_features=self.n_features,
                                max_thresholds=self.max_thresholds)

            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        print("Random Forest training complete.")

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        y_pred = [mode(sample_preds, keepdims=True)[0][0] for sample_preds in tree_predictions]
        return np.array(y_pred)

    def predict_proba(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, 10))
        for i in range(n_samples):
            votes = tree_predictions[i]
            class_votes, counts = np.unique(votes, return_counts=True)
            proba[i, class_votes] = counts / self.n_trees
        return proba


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        print(f"Fitting KNN (memorizing data)...")
        self.X_train = X
        self.y_train = y
        print("Fit complete.")

    def _find_neighbors(self, x_test_row):
        distances = np.sum((self.X_train - x_test_row)**2, axis=1)
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_nearest_indices]
        return mode(k_nearest_labels, keepdims=True)[0][0]

    def predict(self, X_test):
        print(f"Predicting with KNN (k={self.k})... This may be slow.")
        y_pred = [self._find_neighbors(x_row) for x_row in X_test]
        print("KNN prediction complete.")
        return np.array(y_pred)

    def predict_proba(self, X_test):
        print(f"Predicting proba with KNN (k={self.k})... This may be slow.")
        n_samples = X_test.shape[0]
        proba = np.zeros((n_samples, 10))

        for i, x_row in enumerate(X_test):
            distances = np.sum((self.X_train - x_row)**2, axis=1)
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]
            class_votes, counts = np.unique(k_nearest_labels, return_counts=True)

            proba[i, class_votes] = counts / self.k

        print("KNN proba prediction complete.")
        return proba
