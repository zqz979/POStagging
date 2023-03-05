from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from utils import *
import joblib
from matplotlib import pyplot as plt

train_set = load_data('./data/train.txt','dictionary')
test_set = load_data('./data/test.txt','dictionary')

X_train = extract_features(train_set)
y_train = extract_labels(train_set)
X_test = extract_features(test_set)
y_test = extract_labels(test_set)

# Vectorize features
# converting the features to numerical vectors
vectorizer = joblib.load('./models/best_vectorizer.sav')
# all_data=[feature for sentence in X_train for feature in sentence]
# all_data.extend([feature for sentence in X_test for feature in sentence])
# vectorizer = vectorizer.fit(all_data)
X_train_vec = vectorizer.transform([feature for sentence in X_train for feature in sentence])
# vectorizer already fit
X_test_vec = vectorizer.transform([feature for sentence in X_test for feature in sentence])
# shape is (211727, 28268)
# Train model
# 
# clf = joblib.load('./models/mlp_10_1000.pth')
# plt.plot(clf.loss_curve_)
# plt.show()
clf = MLPClassifier(hidden_layer_sizes=(300,), max_iter = 200, early_stopping = True, verbose=True)
clf.fit(X_train_vec, [label for sentence in y_train for label in sentence])
joblib.dump(clf, './models/mlp_(300).pth')

# Test model
y_pred = clf.predict(X_test_vec)

# Evaluate model
accuracy = accuracy_score([label for sentence in y_test for label in sentence], y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

