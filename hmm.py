import utils
from nltk.tag import HiddenMarkovModelTrainer
from sklearn.model_selection import train_test_split

train_sents=utils.load_data()
train_sents, test_sents = train_test_split(train_sents, test_size=0.2, random_state=42)

# Train an HMM on the train set
trainer = HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train_sents)

# Test the HMM tagger on the test set
accuracy = tagger.accuracy(test_sents)

print(f'Testing accuracy: {accuracy:.2%}')