from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# Trains a Decision Tree model and evaluates its performance
def trainModel(data_dict: dict, max_leaf_nodes = None, max_depth = None):
    # Initialize the Decision Tree Classifier with optional max_depth and max_leaf_nodes parameters
    model = DecisionTreeClassifier(random_state=42, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
    
    # Extract training and validation sets from the input dictionary
    inputs_train = data_dict['inputs_train']
    inputs_val = data_dict['inputs_val']
    targets_train = data_dict['targets_train']
    targets_val = data_dict['targets_val']

    # Train the model using the training data
    model.fit(inputs_train, targets_train)
    
    # Generate predictions on the training set and compute the ROC AUC score for training
    train_pred = model.predict(inputs_train)
    train_score = roc_auc_score(targets_train, train_pred)

    # Generate predictions on the validation set and compute the ROC AUC score for validation
    val_pred = model.predict(inputs_val)
    val_score = roc_auc_score(targets_val, val_pred)

    # Return the trained model, training score, and validation score
    return model, train_score, val_score