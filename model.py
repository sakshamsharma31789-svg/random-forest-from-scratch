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

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # TODO: decide whether to stop growing based on purity, depth, and size...
    is_pure = len(np.unique(labels)) <=1
    d = depth>=max_depth
    q = len(labels) < min_samples_split
    return is_pure or d or q
    pass

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    # TODO: choose a single class label to output for a leaf given the labels that reached it
    labels = np.array(labels,dtype=int)
    count = np.bincount(labels)
    max = np.argmax(count)
    return int(max)

    pass

# Step 7 - build_tree
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {'leaf': True, 'prediction': int(leaf_prediction(labels))}
    
    # 2. Determine candidate features
    if feature_subset is None:
        candidate_features = list(range(features.shape[1]))
    else:
        candidate_features = list(feature_subset)
        
    # 3. Find the best split
    split = best_split(features, labels, candidate_features)
    
    # Check if a valid split was found
    if split['feature_index'] is None:
        return {'leaf': True, 'prediction': int(leaf_prediction(labels))}
    
    feature_index = split['feature_index']
    threshold = split['threshold']
    
    # 4. Partition the dataset using helper function
    left_X, left_y, right_X, right_y = split_dataset(features, labels, feature_index, threshold)
    
    # 5. Fallback to leaf if either side is empty
    if len(left_y) == 0 or len(right_y) == 0:
        return {'leaf': True, 'prediction': int(leaf_prediction(labels))}
    
    # 6. Recurse on children
    left_child = build_tree(
        left_X, left_y, 
        max_depth=max_depth, min_samples_split=min_samples_split, 
        feature_subset=feature_subset, depth=depth + 1
    )
    right_child = build_tree(
        right_X, right_y, 
        max_depth=max_depth, min_samples_split=min_samples_split, 
        feature_subset=feature_subset, depth=depth + 1
    )
    
    return {
        'leaf': False,
        'feature_index': int(feature_index),
        'threshold': float(threshold),
        'left': left_child,
        'right': right_child
    }

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    # TODO: walk the example down the fitted tree until you reach a leaf, then return its prediction.
    if tree.get('leaf',False):
        return int(tree['prediction'])
    feature_idx = tree['feature_index']
    threshold = tree['threshold']
    if example[feature_idx] <= threshold:
        return predict_example_tree(tree['left'],example)
    else:
        return predict_example_tree(tree['right'],example)
    pass

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    
    # TODO: return predicted class for each row of features using the fitted tree.
    preds = [predict_example_tree(tree,row) for row in features]
    return np.array(preds,dtype=int)
    pass

# Step 10 - bootstrap_sample
def bootstrap_sample(features, labels, rng):
    # TODO: draw a bootstrap sample of rows (with replacement) using rng.
    n,m = features.shape
    inidces = rng.integers(0,n,size=n)
    return features[inidces],labels[inidces]

    pass

# Step 11 - feature_subset
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    # TODO: return num_to_pick distinct random feature indices from range(num_features) using rng.
    return rng.choice(num_features,size=num_to_pick,replace=False)

    pass

# Step 12 - train_forest
import numpy as np

def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    rng = np.random.default_rng(random_state)
    n_samples, n_features = features.shape
    
    if num_features_per_split is None:
        num_features_per_split = max(1, int(round(np.sqrt(n_features))))
        
    forest = []
    
    for _ in range(num_trees):
        boot_x, boot_y = bootstrap_sample(features, labels, rng)
        
        feat_indices = feature_subset(n_features, num_features_per_split, rng)
        
        tree = build_tree(
            boot_x, 
            boot_y, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            feature_subset=feat_indices
        )
        
        forest.append({
            'tree': tree,
            'feature_indices': feat_indices
        })
        
    return forest
    pass

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

