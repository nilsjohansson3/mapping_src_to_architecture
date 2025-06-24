
import numpy as np
from collections import Counter

class MethodClassifier:
    def __init__(self, clf):
        self.clf = clf
    
    def predict(self, X, method='method_content', method_id_of_identifiers=None, use_prob_threshold=False):
        if method == 'method_content':
            return self.predict_method_content(X)
        elif method == 'most_common_identifier':
            if method_id_of_identifiers is None:
                raise ValueError("method_id_of_identifiers must be provided for 'most_common_identifier' method")
            return self.predict_most_common_identifier(X, method_id_of_identifiers)
        elif method == 'summarized_probabilities':
            if method_id_of_identifiers is None:
                raise ValueError("method_id_of_identifiers must be provided for 'summarized_probabilities' method")
            return self.predict_summarized_probabilities(X, method_id_of_identifiers, use_prob_threshold)
        elif method == 'top3_identifiers':
            if method_id_of_identifiers is None:
                raise ValueError("method_id_of_identifiers must be provided for 'top3_identifiers' method")
            return self.predict_top3_identifiers(X, method_id_of_identifiers)
        else:
            raise ValueError(f"Invalid method: {method}")
        
    def predict_method_content(self, X, method_ids_all, N, use_prob_threshold=False):
        used_method_ids = []
        for method_id in method_ids_all:
            used_method_ids.extend([method_id])

        # Get predictions from the classifier
        predictions = self.clf.predict(X, N, use_prob_threshold)

        # Check if predictions are in tuple format
        if isinstance(predictions, tuple):
            y_pred_top_n = predictions[0]  # Extract top N predictions
        else:
            y_pred_top_n = predictions

        # Ensure y_pred_top_n is consistently shaped for conversion to NumPy array
        try:
            y_pred_top_n_array = np.array(y_pred_top_n)
        except Exception as e:
            print(f"Error converting predictions to array: {e}")
            y_pred_top_n_array = np.array([])  # Default to empty array in case of error

        #X_eval_entropy = self.get_neighbor_entropy(X_train_embeddings, y_train_encoded, X)

        #X_eval_density = self.get_neighbor_density(X_train_embeddings, X)

        return y_pred_top_n_array, used_method_ids#, X_eval_entropy, X_eval_density

    def predict_summarized_probabilities(self, X, method_id_of_identifiers, N=1, use_prob_threshold=True):
        # Get predicted probabilities from the classifier
        y_pred_identifiers_encoded_proba = self.clf.predict_proba(X, N=N, useProbThreshold=use_prob_threshold)
        
        y_pred_topN = []
        method_ids_topN = []

        unique_method_ids = np.unique(method_id_of_identifiers)

        for method_idx in unique_method_ids:
            # Get indices for the current method_id
            method_indices = np.where(method_id_of_identifiers == method_idx)[0]
            
            # Aggregate probabilities across all data points with the same method_id
            aggregated_proba = y_pred_identifiers_encoded_proba[method_indices].sum(axis=0)
            
            # Get the indices of the top N classes
            topN_indices = np.argsort(aggregated_proba)[-N:][::-1]
            
            # Append the top N predictions for this method_id
            y_pred_topN.extend(topN_indices.reshape(1, -1).tolist())
            method_ids_topN.extend([method_idx])  # Store method_idx for each top N prediction group

        y_pred_topN = np.array(y_pred_topN)  # Convert list to an array

        return y_pred_topN, method_ids_topN

    def predict_top3_identifiers(self, X, method_id_of_identifiers):
        if not hasattr(self.clf, 'predict_proba'):
            raise ValueError("Classifier does not support probability predictions")
        
        y_pred_identifiers_encoded_proba = self.clf.predict_proba(X)
        y_pred_top3 = []
        method_ids_top3 = []

        for method_idx in np.unique(method_id_of_identifiers):
            method_indices = np.where(method_id_of_identifiers == method_idx)[0]
            top3_indices = np.argsort(y_pred_identifiers_encoded_proba[method_indices], axis=1)[:, -3:][:, ::-1]
            
            # Flatten and append the top 3 predictions for all samples corresponding to the method_idx
            y_pred_top3.extend(top3_indices.tolist())
            method_ids_top3.extend([method_idx] * len(method_indices))  # Replicate method_idx for each sample in the group

        y_pred_top3 = np.array(y_pred_top3)  # Convert list of lists to an n x 3 NumPy array
        
        return y_pred_top3, method_ids_top3
    
    def get_neighbor_entropy(self, X_train, y_train, X_test):
        results, mean_entropy = self.clf.compute_neighbor_label_entropy_all(X_train, y_train, X_test)
        return [entry['entropy'] for entry in results], mean_entropy
    
    def get_neighbor_density(self, X_train, X_test):
        results, mean_density = self.clf.compute_neighborhood_density(X_train, X_test)
        return results, mean_density


