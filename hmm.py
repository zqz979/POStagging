from utils import *
from nltk.tag import HiddenMarkovModelTrainer, HiddenMarkovModelTagger
from sklearn.model_selection import train_test_split
def preprocess():
    train_sents=load_data('./data/train.txt', 'tuple')
    train_sents, test_sents = train_test_split(train_sents, test_size=0.2)
    return train_sents, test_sents

def train_model(train_sents):
    # Train an HMM on the train set
    trainer = HiddenMarkovModelTrainer()
    model = trainer.train_supervised(train_sents)

    # Create an HMM tagger from the trained model
    tagger = HiddenMarkovModelTagger(model._symbols, model._states, model._transitions, model._outputs, model._priors)

    return tagger

def test_model(model,test_sents):
    # Test the HMM tagger on the test set
    accuracy = model.accuracy(test_sents)
    print(f'Testing accuracy: {accuracy:.2%}')

def main():
    train_sents, test_sents = preprocess()
    print(train_sents[0])
    test_model(train_model(train_sents),test_sents)

if __name__ == "__main__":
    main()