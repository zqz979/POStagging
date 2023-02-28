from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from utils import *

def main():

    train_data = load_data('./data/train.txt','dictionary')
    train_set, test_set = train_test_split(train_data, test_size=0.2)
    # training the model using entire train.txt and evaluate model on the actual test.txt, the accuracy is about 97%
    X_train = extract_features(train_set)
    y_train = extract_labels(train_set)
    X_test = extract_features(test_set)
    y_test = extract_labels(test_set)

    # Vectorize features
    # converting the features to numerical vectors
    vectorizer = DictVectorizer()
    t1=[feature for sentence in X_train for feature in sentence]
    t1.extend([feature for sentence in X_test for feature in sentence])
    vectorizer = vectorizer.fit(t1)
    X_train_vec = vectorizer.transform([feature for sentence in X_train for feature in sentence])
    # vectorizer already fit
    X_test_vec = vectorizer.transform([feature for sentence in X_test for feature in sentence])

    
    # Train model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, [label for sentence in y_train for label in sentence])

    # Test model
    y_pred = clf.predict(X_test_vec)

    # Evaluate model
    accuracy = accuracy_score([label for sentence in y_test for label in sentence], y_pred)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

if __name__ == "__main__":
    main()