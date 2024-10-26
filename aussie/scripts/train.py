from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
def trainModel(data_dict: dict, max_leaf_nodes = None, max_depth = None):
    model = DecisionTreeClassifier(random_state = 42, max_depth = max_depth, max_leaf_nodes = max_leaf_nodes)
    inputs_train = data_dict['inputs_train']
    inputs_val = data_dict['inputs_val']
    targets_train = data_dict['targets_train']
    targets_val = data_dict['targets_val']

    model.fit(inputs_train, targets_train)
    train_pred = model.predict(inputs_train)
    train_score = roc_auc_score(targets_train, train_pred)

    val_pred = model.predict(inputs_val)
    val_score = roc_auc_score(targets_val, val_pred)

    #print("Max Depth {} Max Leaf nodes {} Train score id {} and validation score is {}".format(max_depth, max_leaf_nodes, train_score, val_score))
    return model, train_score, val_score