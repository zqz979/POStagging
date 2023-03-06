from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from utils import *
from matplotlib import pyplot as plt

train_set = load_data('./data/train.txt', 'dictionary')
test_set = load_data('./infer/test.txt', 'dictionary', True)
eval_set = load_data('./data/test.txt', 'dictionary')

X_train = extract_features(train_set)
y_train = extract_labels(train_set)
X_test = extract_features(test_set)

vectorizer = DictVectorizer()
t1 = [feature for sentence in X_train for feature in sentence]
t1.extend([feature for sentence in X_test for feature in sentence])
vectorizer = vectorizer.fit(t1)
X_train_vec = vectorizer.transform(
    [feature for sentence in X_train for feature in sentence])
X_test_vec = vectorizer.transform(
    [feature for sentence in X_test for feature in sentence])
X_train_vec=Normalizer().transform(X_train_vec)
X_test_vec=Normalizer().transform(X_test_vec)

clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=10, verbose=True)
clf.fit(X_train_vec, [label for sentence in y_train for label in sentence])

# Test model
y_pred = clf.predict(X_test_vec)

# Evaluate model
y_eval = extract_labels(eval_set)
accuracy = accuracy_score(
    [label for sentence in y_eval for label in sentence], y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

save_model(clf, "mlp.pth")
save_model(vectorizer, "vectorizer.sav")