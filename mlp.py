from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import *
from matplotlib import pyplot as plt

train_data = load_data('./data/train.txt', 'dictionary')
test_set = load_data('./data/test.txt', 'dictionary')

train_set, eval_set = train_test_split(train_data, test_size=0.2)

X_train = extract_features(train_set)
y_train = extract_labels(train_set)
X_eval = extract_features(eval_set)
y_eval = extract_labels(eval_set)
X_test = extract_features(test_set)

vectorizer = DictVectorizer()
t1 = [feature for sentence in X_train for feature in sentence]
t1.extend([feature for sentence in X_test for feature in sentence])
t1.extend([feature for sentence in X_eval for feature in sentence])
vectorizer = vectorizer.fit(t1)
X_train_vec = vectorizer.transform(
    [feature for sentence in X_train for feature in sentence])
X_eval_vec = vectorizer.transform(
    [feature for sentence in X_eval for feature in sentence])

clf = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=50, verbose=True)
clf.fit(X_train_vec, [label for sentence in y_train for label in sentence])

# Test model
y_pred = clf.predict(X_eval_vec)

# Evaluate model
accuracy = accuracy_score(
    [label for sentence in y_eval for label in sentence], y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

save_model(clf,"./models/mlp.pth")
save_model(vectorizer,"./models/vectorizer.sav")