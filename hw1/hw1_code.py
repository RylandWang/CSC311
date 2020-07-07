# real vs fake news classifier
# using decision tree and k-nearest-neighbor
import sklearn
import math
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import *
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

# file paths for data
FAKE_HEADLINES = "data/clean_fake.txt"
REAL_HEADLINES = "data/clean_real.txt"
# ratio for splitting data set
TRAIN_SPLIT, TEST_SPLIT, VALIDATION_SPLIT = 0.7, 0.15, 0.15
# headline labels
FAKE_LABEL, REAL_LABEL = "fake", "real"

def load_data(data_set: list) -> (list, list):
    '''
    Preprocess raw data set using vectorizer and split the resulting matrix into 3 sets for: training, validating, and testing
    '''

    headlines = [x for (x,t) in data_set]
    labels = [t for (x,t) in data_set]

    # vectorize headlines into M x N sparse matrix
    # where M = # of headlines, and N = total # number of words
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(headlines)
    X_feature_names = vectorizer.get_feature_names()

    # labels are M x 2
    y = vectorizer.fit_transform(labels)
    y_class_names = vectorizer.get_feature_names()

    # randomize dataset order and split into training, test, and validation sets
    train_set, t_and_v_set, train_labels, t_and_v_labels = train_test_split(X.toarray(), y.toarray(), test_size = 1-TRAIN_SPLIT)
    test_set, validation_set, test_labels, validation_labels = train_test_split(t_and_v_set, t_and_v_labels, test_size = (VALIDATION_SPLIT/(TEST_SPLIT+VALIDATION_SPLIT)))

    return train_set, validation_set, test_set, train_labels, validation_labels, test_labels, X_feature_names, y_class_names

def select_tree_model(train_set, validation_set, train_labels, validation_labels) -> DecisionTreeClassifier:
    '''
    Evaluate validation accuracies of different decision tree models by varying the following hyperparameters: 
        1. max depth
        2. split criterion: gini vs entropy
    '''
    best_accuracy = 0
    best_model = None

    # finetune hyperparameters and select ones that achieve highest accuracy
    for depth in [3, 5, 15, 50, 125]:
        for criteria in ["gini", "entropy"]:
            classifier = DecisionTreeClassifier(max_depth=depth, criterion=criteria)
            # train model on training data
            classifier.fit(train_set, train_labels)

            # test accuracy score on validation set
            accuracy = evaluate_model(classifier, validation_set, validation_labels)
            print(f"Criterion = {criteria}, Depth = {depth} => Accuracy score = {accuracy}")

            if accuracy > best_accuracy:
                best_accuracy, best_model = accuracy, classifier

    print("Best accuracy achieved on validation set: ", best_accuracy)
    return best_model


def select_knn_model(train_set, validation_set, train_labels, validation_labels):
    '''
    Evaluate validation accuracies of different k-nearest-neighbor models by varying the following hyperparameters: k.
    Visualize error rates on graph for each k
    '''
    best_accuracy = 0
    best_model = None
    k_values, train_error, validation_error = [], [], []

    # test model accuracy on different values of hyperparameter k
    for k in range(1, 21):
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(train_set, train_labels)
        
        # k values independant variables 
        k_values.append(k)

        # record error rate on training set
        train_accuracy = classifier.score(train_set, train_labels)
        train_error.append(1-train_accuracy)

        # record error rate on validation set
        validation_accuracy = classifier.score(validation_set, validation_labels)
        validation_error.append(1-validation_accuracy)
        print(f"K = {k} => Accuracy score = {validation_accuracy}")

        if validation_accuracy > best_accuracy:
            best_accuracy, best_model = validation_accuracy, classifier
    
    print("Best accuracy achieved on validation set: ", best_accuracy)

    # plot error rates on graph

    plt.plot(k_values, train_error, label="Train")
    plt.plot(k_values, validation_error, label="Validation")
    plt.legend(loc="upper left")
    plt.xlabel("k (# of nearest neighbors)")
    plt.ylabel("Error rate")
    plt.show()

    return best_model

def evaluate_model(model: DecisionTreeClassifier, data, labels) -> float:
    '''
    Run predictions on data set using model, and compare predictions against labels(target set) to compute the model accuracy
    '''
    total_correct = 0
    for i, x in enumerate(model.predict(data)):
        total_correct += (1 - abs(labels[i] - x))[0]
    return total_correct/len(data)

def _p_log_p(probability_x):
    if probability_x == 0:
        return 0
    return probability_x * math.log(probability_x, 2)

def root_entropy(labels):
    """
    computes H(Y), where Y is the random variable
    """
    entropy = 0
    labels_count = dict()
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            labels_count[j] = labels_count.get(j, 0) + labels[i][j]
    
    # summation p(x)*log(p(x))
    for label in labels_count:
        entropy -= _p_log_p(labels_count[label]/len(labels))

    return entropy

def leafs_entropy(key_word, data, labels, feature_names):
    '''
    computes H(Y|B), where Y is random variable, and B is split direction on "key_word"
    '''
    entropy, l_entropy, r_entropy = 0, 0, 0
    split_true = [0, 0] # [fake, real]
    split_false = [0, 0]

    # perform split on rows
    # by scanning each column for keyword
    for i in range(len(data)): # row
        found_word = False
        for j in range(len(data[i])): # col
            if data[i][j]:
                if feature_names[j] == key_word:
                    split_true += labels[i]
                    found_word = True
                    break
        if not found_word:
            split_false += labels[i]

    # now compute total entropy after split
    # H(Y|B) = H(Y |B =left)P(B =left) + H(Y |B = right)P(B = right)
    if sum(split_true):
        for label_count in split_true:
            l_entropy -= _p_log_p(label_count/sum(split_true))
    
    if sum(split_false):
        for label_count in split_false:
            r_entropy -= _p_log_p(label_count/sum(split_false))

    return ((sum(split_false)* l_entropy) + (sum(split_true) * r_entropy)) / len(labels)


def compute_information_gain(key_word, data, labels, feature_names):
    '''
    Compute information gain for splitting on a particular word
    IG(Y|B) = H(Y) - H(Y|B)
    '''
    return root_entropy(labels) - leafs_entropy(key_word, data, labels, feature_names)


if __name__ == "__main__":

    data_set = []
    for headline in open(FAKE_HEADLINES, "r").readlines():
        data_set.append((headline, FAKE_LABEL))

    for headline in open(REAL_HEADLINES, "r").readlines():
        data_set.append((headline, REAL_LABEL))

    train_set, validation_set, test_set, train_labels, validation_labels, test_labels, X_feature_names, y_class_names = load_data(data_set)   

    tree_classifier = select_tree_model(train_set, validation_set, train_labels, validation_labels)
    print("Tree prediction accuracy on test set: ", tree_classifier.score(test_set, test_labels))

    # create graph visualizing the classifier
    tree.export_graphviz(tree_classifier, out_file="tree.dot", feature_names=X_feature_names, max_depth=2)
    
    print()
    knn_model = select_knn_model(train_set, validation_set, train_labels, validation_labels)
    print("Knn prediction accuracy on test set: ", knn_model.score(test_set, test_labels))
