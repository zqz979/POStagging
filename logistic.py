from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from utils import *

# returns data which contains a list of sentences, 
# where each sentence is a list of dictionaries representing the words in that sentence along with their POS tags.
def load_data(filename):
    # use to store the list of sentences
    data = []
    with open(filename, 'r') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if not line:  # empty line indicates end of sentence
                data.append(sentence)
                sentence = []
            else:
                # last column is ignore
                word, pos_tag, _ = line.split()
                sentence.append({'word': word, 'pos_tag': pos_tag})
        if sentence:
            data.append(sentence)
    return data

# takes in a list of sentences and extracts features for each word in each sentence
def extract_features(sentences):
    return [[feature_extractor(sentence, i) for i in range(len(sentence))] for sentence in sentences]

# takes in a list of sentences and extracts the part-of-speech tags for each token in each sentence
# The resulting part-of-speech tags are stored in a list of lists where each outer list represents a sentence 
# and each inner list contains the part-of-speech tags for the tokens in that sentence
def extract_labels(sentences):
    return [[token['pos_tag'] for token in sentence] for sentence in sentences]

def main():

    train_data = load_data('./data/train.txt')
    train_set, test_set = train_test_split(train_data, test_size=0.2)
    # training the model using entire train.txt and evaluate model on the actual test.txt, the accuracy is about 97%
    #test_data = load_data('./data/test.txt')
    X_train = extract_features(train_set)
    y_train = extract_labels(train_set)
    X_test = extract_features(test_set)
    y_test = extract_labels(test_set)

    # Vectorize features
    # converting the features to numerical vectors
    vectorizer = DictVectorizer()
    X_train_vec = vectorizer.fit_transform([feature for sentence in X_train for feature in sentence])
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