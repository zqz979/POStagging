import gensim.downloader as api
import string
import pickle
import torch
from embed_def import *
from nltk.tag import HiddenMarkovModelTrainer, HiddenMarkovModelTagger

# returns data which contains a list of sentences,
# option to load as list of dictionaries or tuple
# e.g. {'word': 'in', 'pos_tag': 'IN'} or ('in', 'IN')
def load_data(filename, format):
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
                fields = line.split()
                if(len(fields)<2):
                    fields=(fields[0],"")
                if (format == 'dictionary'):
                    sentence.append({'word': fields[0], 'pos_tag': fields[1]})
                if (format == 'tuple'):
                    sentence.append((fields[0],fields[1]))
        if sentence:
            data.append(sentence)
    return data

def write_infer(filename, inputs, preds):
    with open(filename, 'w') as f:
        i = 0
        for sentence in inputs:
            for word in sentence:
                f.write(f'{word["word"]} {preds[i]}\n')
                i += 1
            f.write('\n')

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

pre_glob_model = api.load("glove-twitter-100")
def cust_emb():
    cust_model=load_model("./custom_embeddings/mlp.pth")
    X_enc=load_model("./custom_embeddings/X_enc.sav")
    for name, module in cust_model.named_modules():
        if(name=="embed"):
            break
    model={}
    for i in range(module.num_embeddings):
        key=X_enc.inverse_transform([i])[0]
        val=module(torch.LongTensor([i])).tolist()[0]
        model[key]=val
    return model
cust_glob_model=cust_emb()

def word2emb(word):
    try:
        emb = pre_glob_model.get_vector(word)
    except:
        emb = pre_glob_model.get_vector("unk")
    cust_emb=cust_glob_model.get(word, next(iter(cust_glob_model.values())))
    return emb,cust_emb

def get_hmm():
    train_sents=load_data('./data/train.txt', 'tuple')
    # Train an HMM on the train set
    trainer = HiddenMarkovModelTrainer()
    model = trainer.train_supervised(train_sents)
    # Create an HMM tagger from the trained model
    tagger = HiddenMarkovModelTagger(model._symbols, model._states, model._transitions, model._outputs, model._priors)
    model={}
    for sent in train_sents:
        for word in sent:
            val=tagger.tag([word])[0][1]
            model[word]=val
    return model

glob_hmm=get_hmm()
def word2hmm(word):
    tag=glob_hmm.get(word, next(iter(glob_hmm.values())))
    return tag

def is_punc(word):
    if word in string.punctuation:
        return True
    return False

def num_punc(word):
    cnt = 0
    for c in word:
        if (c in string.punctuation):
            cnt += 1
    return cnt

def num_caps(word):
    cnt = 0
    for c in word:
        if (c.isupper()):
            cnt += 1
    return cnt

def num_digit(word):
    cnt = 0
    for c in word:
        if (c.isdigit()):
            cnt += 1
    return cnt

def num_dash(word):
    return word.count('-')

def num_periods(word):
    return word.count('.')

def perc_cap_letters(word):
    cnt = 0
    cntC = 0
    for c in word:
        if (c.isalpha()):
            cnt += 1
            if (c.isupper()):
                cntC += 1
    if (cnt == 0):
        return 0
    return cntC/cnt

def num_slashes(word):
    cnt = 0
    for i in range(len(word)):
        if (word[i] == '\\'):
            if (i == len(word)-1):
                continue
            elif (word[i+1] == '/'):
                cnt += 1
    return cnt

# Get features for each sentence
def feature_extractor(sentence, i):
    curr_word = sentence[i]['word']
    if (i == 0):
        prev_word = '<s>'
    else:
        prev_word = sentence[i-1]['word']
    if (i == len(sentence)-1):
        next_word = '</s>'
    else:
        next_word = sentence[i+1]['word']
    if (i <= 1):
        prev_prev_word = '<s>'
    else:
        prev_prev_word = sentence[i-2]['word']
    if (i >= len(sentence)-2):
        next_next_word = '</s>'
    else:
        next_next_word = sentence[i+2]['word']
    features = {}
    for w, word in enumerate([curr_word, prev_word, next_word, prev_prev_word, next_next_word]):
        features.update({
            # suffixes
            f'{w}_suffix1': word[-3:],
            f'{w}_suffix2': word[-2:],
            f'{w}_suffix3': word[-1:],
            # prefixes
            f'{w}_prefix1': word[:3],
            f'{w}_prefix2': word[:2],
            f'{w}_prefix3': word[:1],
            # All-cap word can be proper nouns or abbreviations
            f'{w}_word.isupper()': word.isupper(),
            # Capitalized words can be beginning of sentences, proper nouns, and acronyms.
            f'{w}_word.istitle()': word.istitle(),
            # if the word is a number
            f'{w}_word.isdigit()': word.isdigit(),
            f'{w}_punc': is_punc(word),
            f'{w}_len': len(word),
            f'{w}_end': i == len(sentence)-1 or word == "</s>",
            f'{w}_start': i == 0 or word == "<s>",
            f'{w}_ind': i,
            f'{w}_numpunc': num_punc(word),
            f'{w}_numcaps': num_caps(word),
            f'{w}_numdigit': num_digit(word),
            f'{w}_numdash': num_dash(word),
            f'{w}_numperiod': num_periods(word),
            f'{w}_perccapletters': perc_cap_letters(word),
            f'{w}_numslashes': num_slashes(word),
            f'{w}_relloc': i/len(sentence)
        })
        emb,cust_emb=word2emb(word)
        for i, v in enumerate(emb):
            features[f'{w}_pre_emb_{i}'] = v
        for i, v in enumerate(cust_emb):
            features[f'{w}_cust_emb_{i}'] = v
        features[f'{w}_hmm']=word2hmm(word)
    return features

# get features for entire data
def extract_features(sentences):
    return [[feature_extractor(sentence, i) for i in range(len(sentence))] for sentence in sentences]

# get labels for each training data
def extract_labels(sentences):
    return [[token['pos_tag'] for token in sentence] for sentence in sentences]