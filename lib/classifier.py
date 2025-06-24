import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
import math

def dot_product_distance(x, y):
    return -np.dot(x, y)

class Classifier:
    def __init__(self, method, hyperparameters):
        self.method = method
        self.hyperparameters = hyperparameters
        self.n_neighbors = -1
        self.model = None
        self.tracked_labels = {}
        self.min_class_size = -1

    def train(self, X, y):
        class_counts = Counter(y)
        self.min_class_size = min(class_counts.values())
        if self.method == 'kNN':
            # Define the KNN classifier
            n_neighbors = self.hyperparameters['n_neighbors']
            weights = self.hyperparameters['weights']
            
            # Handle the "adaptive" option for n_neighbors
            if n_neighbors == "sqrt_min_class":
                #class_counts = Counter(y)
                unique_classes, counts = np.unique(y, return_counts=True)
                smallest_class_size = min(counts)
                k = int(math.sqrt(smallest_class_size))# * len(unique_classes) 
                n_neighbors = max(1, k)  # Ensure at least 1 neighbor
                self.n_neighbors = n_neighbors
            elif n_neighbors == 'min_class':
                class_counts = Counter(y)
                n_neighbors = min(class_counts.values())

            # Ensure n_neighbors is odd
            if n_neighbors % 2 == 0:
                n_neighbors -= 1

            knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric='cosine') # , metric = 'cosine', 'manhattan'
            # Create a pipeline with a scaler and the KNN classifier
            pipeline = make_pipeline(StandardScaler(), knn)

            # Train the KNN classifier
            self.model = pipeline.fit(X, y)

    def predict(self, X, N=1, useProbThreshold=False):
        """
        Predict the top N classes for each data point in X.

        :param X: Input data.
        :param N: Number of top classes to return for each data point.
        :param useProbThreshold: If True, apply a probability threshold for predictions.
        :return: The top N classes and their probabilities (if N > 1).
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Please call train() first.")
        
        if self.method == 'kNN':
            probabilities = self.model.predict_proba(X)

            if useProbThreshold:
                threshold = self.hyperparameters['threshold']
                above_threshold = probabilities.max(axis=1) >= threshold
                predictions = np.where(above_threshold, probabilities.argmax(axis=1), -1)
                return predictions if N == 1 else (self._get_top_n_classes(probabilities, N))
            else:
                return self._get_top_n_classes(probabilities, N)
            
    def predict_proba(self,X, N=1, useProbThreshold=False):
        if self.model is None:
            raise ValueError("Model has not been trained. Please call train() first.")
        
        if self.method == 'kNN':
            probabilities = self.model.predict_proba(X)

            if useProbThreshold:
                threshold = self.hyperparameters['threshold']
                above_threshold = probabilities >= threshold
                probabilities[~above_threshold] = 0
                ## ALL BELOW THRESHOLD SHALL BE SET TO -1 IN THE PROBABILITIES ARRAY
                ## THEN RETURN PROBABILITIES
                return probabilities
            else:
                return probabilities
            
    def compute_neighbor_label_entropy_all(self, X_train, y_train, X_test):
        """
        Compute entropy of neighbor label distribution for all test points.

        Returns:
            List of (index, entropy, neighbor_labels)
        """

        k = self.min_class_size
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine')
        nbrs.fit(X_train)

        results = []
        entropies = []

        for i, x in enumerate(X_test):
            distances, indices = nbrs.kneighbors([x])
            neighbor_labels = [y_train[idx] for idx in indices[0]]
            
            label_counts = Counter(neighbor_labels)
            probs = np.array(list(label_counts.values())) / k
            ent = entropy(probs, base=2)
            entropies.append(ent)
            results.append({
                'index': i,
                'entropy': ent,
                'neighbor_labels': neighbor_labels
            })
        mean_entropy = np.mean(entropies) if entropies else 0.0
        return results, mean_entropy


    def compute_neighborhood_density(self, X_train, X_test):
        """
        Compute average cosine similarity to k nearest training points for each test point.
        
        Returns:
            List of dicts with test index, average similarity, and similarity vector
        """
        k = self.min_class_size
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine')
        nbrs.fit(X_train)

        avg_similarities = []

        for x in X_test:
            distances, _ = nbrs.kneighbors([x])
            sims = 1 - distances[0]  # cosine similarity = 1 - distance
            avg_sim = np.mean(sims)
            avg_similarities.append(avg_sim)
        mean_similarity = np.mean(avg_similarities) if avg_similarities else 0.0
        return avg_similarities, mean_similarity


    def _get_top_n_classes(self, probabilities, N):
        """
        Get the top N classes and their probabilities for each data point.

        :param probabilities: Array of predicted probabilities.
        :param N: Number of top classes to return.
        :return: Top N classes and their probabilities.
        """
        if N == 1:
            return probabilities.argmax(axis=1), None  # Return None for probabilities
        else:
            top_n_indices = np.argsort(probabilities, axis=1)[:, -N:][:, ::-1]
            top_n_probabilities = np.take_along_axis(probabilities, top_n_indices, axis=1)
            return top_n_indices, top_n_probabilities
