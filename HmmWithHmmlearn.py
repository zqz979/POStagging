from utils import *
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


train_sents = load_data('./data/train.txt', 'tuple')
train_sents, test_sents = train_test_split(train_sents, test_size=0.2)


words = []
tags = []
train_sents_code = []
test_sents_code = []

#Give code to the tags and the words under the 1-d array with all of the words/tags together and the marks of changing line
word_encoder = LabelEncoder()
tag_encoder = LabelEncoder()
words.append("###")
tags.append("###")
for sentence in train_sents:
    for word, tag in sentence:
        words.append(word)
        tags.append(tag) 
    words.append("###")
    tags.append("###")
X = word_encoder.fit_transform(words)
y = tag_encoder.fit_transform(tags)
c_sen_word = X[0]
c_sen_tag = y[0]
word2num = list(word_encoder.classes_)
tag2num = list(tag_encoder.classes_)
num_word = len(word2num)
num_tag = len(tag2num)
words2 = []
tags2 = []

#Remove the marks and move the code of the words to the original place in the array
flag = True
for word, tag in zip(X,y):
    if flag:
        flag = False
        continue
    if word != c_sen_word and tag != c_sen_tag:
        words2.append(word)
        tags2.append(tag) 
    else:
        sens = list(zip(words2,tags2))
        train_sents_code.append(sens)
        words2 = []
        tags2 = []
        
#Calculate initial state occupation distribution(pi), matrix of transition probabilities(A), and probability of emitting for hmm(B) 
#Also give the useless value to the symbol of changing lines to make sure the model successfully run
#These matrices are calculated under the encode values
pi = np.zeros((num_tag,))
A = np.zeros((num_tag, num_tag))
B = np.zeros((num_tag, num_word))
num_sen = len(train_sents_code)
for sentence in train_sents_code:
    pi[sentence[0][1]] += 1
    term = [(t0[1],t1[1]) for t0,t1 in zip(sentence,sentence[1:])]
    for p1,p2 in term:
        A[p1][p2] += 1
    for w,p in sentence:
        B[p][w] += 1
pi = pi / num_sen
for i in range(len(A)):
    sumab = 0
    for j in A[i]:
        sumab += j
    if sumab != 0:
        A[i] /= sumab
A[c_sen_tag][c_sen_tag] = 1
for i in range(len(B)):
    sumab = 0
    for j in B[i]:
        sumab += j
    if sumab != 0:
        B[i] /= sumab
B[c_sen_tag][c_sen_word] = 1

#Set the parameters of the model with pi,A,B 
hmm_model = hmm.CategoricalHMM(n_components=num_tag, algorithm='viterbi')
hmm_model.startprob_ = pi
hmm_model.transmat_ = A
hmm_model.emissionprob_ = B

#Set the dir from word to code
wordnum_dir = {}
for i in range(len(word2num)):
    wordnum_dir[word2num[i]] = i
tagnum_dir = {}
for i in range(len(tag2num)):
    tagnum_dir[tag2num[i]] = i
    
test_words = []
sen_lens = []
for sentence in test_sents:
    term2 = []
    for word, tag in sentence:
        if word in wordnum_dir:
            term2.append([wordnum_dir[word]])
    if term2 != []:
        test_words.append(term2)

#predict the word in different sentences
y_eval = []
key_predict = []
for sentence in test_words:
    sen_pre = hmm_model.predict(sentence)
    key_predict.append(sen_pre)

#get the accuracy
y_test = []
for sentence in test_sents:
    for word, tag in sentence:
        if word in wordnum_dir:
            y_test.append(tagnum_dir[tag])
for keylist in key_predict:
    for val in keylist:
        y_eval.append(val)
print("The accuracy is {}".format(accuracy_score(y_test, y_eval))) 