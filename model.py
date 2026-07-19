"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # TODO: score how mixed the labels are; 0 for a pure set, larger for more mixed sets.
    if len(labels) <= 1:
        return 0
    _, count = np.unique(labels,return_counts =True)
    probabilities = count / len(labels)
    gini = 1.0 - np.sum(probabilities**2)
    return float(gini)
    pass

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    col = features[:,feature_index]
    mask = (col<=threshold)
    return features[mask],labels[mask],features[~mask],labels[~mask]
    pass

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    n = len(parent_labels)
    w_l = len(left_labels) / n
    w_r = len(right_labels) / n 
    return impurity(parent_labels) - ((w_l * impurity(left_labels)) + (w_r*impurity(right_labels)))
    pass

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    # TODO: search feature_indices for the (feature, threshold) that best improves purity.
    best = {
        'feature_index':None,
        'threshold':None,
        'score':0
    }
    for fi in feature_indices:
        feature_values = features[:,fi]
        unique_values = np.unique(feature_values)
        if len(unique_values) <=1:
            continue
        thresholds = (unique_values[:-1]+unique_values[1:])/2
        for t in thresholds:
            lf,ll,rf,rl = split_dataset(features,labels,fi,t)
            if len(ll) == 0 or len(rl) == 0:
                continue
            score = split_score(labels,ll,rl)
            if score>best['score']:
                best['feature_index']=fi
                best['threshold'] = t
                best['score'] = score
    return best
    pass

# Step 5 - should_stop (not yet solved)
# TODO: implement

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

