import os
import nltk
import gensim.downloader as api
import string
import pickle
import numpy as np

GLOVE_AVG_MISS_STRING_300 = '0.22418134 -0.28881392 0.13854356 0.00365387 -0.12870757 0.10243822 0.061626635 0.07318011 -0.061350107 -1.3477012 0.42037755 -0.063593924 -0.09683349 0.18086134 0.23704372 0.014126852 0.170096 -1.1491593 0.31497982 0.06622181 0.024687296 0.076693475 0.13851812 0.021302193 -0.06640582 -0.010336159 0.13523154 -0.042144544 -0.11938788 0.006948221 0.13333307 -0.18276379 0.052385733 0.008943111 -0.23957317 0.08500333 -0.006894406 0.0015864656 0.063391194 0.19177166 -0.13113557 -0.11295479 -0.14276934 0.03413971 -0.034278486 -0.051366422 0.18891625 -0.16673574 -0.057783455 0.036823478 0.08078679 0.022949161 0.033298038 0.011784158 0.05643189 -0.042776518 0.011959623 0.011552498 -0.0007971594 0.11300405 -0.031369694 -0.0061559738 -0.009043574 -0.415336 -0.18870236 0.13708843 0.005911723 -0.113035575 -0.030096142 -0.23908928 -0.05354085 -0.044904727 -0.20228513 0.0065645403 -0.09578946 -0.07391877 -0.06487607 0.111740574 -0.048649278 -0.16565254 -0.052037314 -0.078968436 0.13684988 0.0757494 -0.006275573 0.28693774 0.52017444 -0.0877165 -0.33010918 -0.1359622 0.114895485 -0.09744406 0.06269521 0.12118575 -0.08026362 0.35256687 -0.060017522 -0.04889904 -0.06828978 0.088740796 0.003964443 -0.0766291 0.1263925 0.07809314 -0.023164088 -0.5680669 -0.037892066 -0.1350967 -0.11351585 -0.111434504 -0.0905027 0.25174105 -0.14841858 0.034635577 -0.07334565 0.06320108 -0.038343467 -0.05413284 0.042197507 -0.090380974 -0.070528865 -0.009174437 0.009069661 0.1405178 0.02958134 -0.036431845 -0.08625681 0.042951006 0.08230793 0.0903314 -0.12279937 -0.013899368 0.048119213 0.08678239 -0.14450377 -0.04424887 0.018319942 0.015026873 -0.100526 0.06021201 0.74059093 -0.0016333034 -0.24960588 -0.023739101 0.016396184 0.11928964 0.13950661 -0.031624354 -0.01645025 0.14079992 -0.0002824564 -0.08052984 -0.0021310581 -0.025350995 0.086938225 0.14308536 0.17146006 -0.13943303 0.048792403 0.09274929 -0.053167373 0.031103406 0.012354865 0.21057427 0.32618305 0.18015954 -0.15881181 0.15322933 -0.22558987 -0.04200665 0.0084689725 0.038156632 0.15188617 0.13274793 0.113756925 -0.095273495 -0.049490947 -0.10265804 -0.27064866 -0.034567792 -0.018810693 -0.0010360252 0.10340131 0.13883452 0.21131058 -0.01981019 0.1833468 -0.10751636 -0.03128868 0.02518242 0.23232952 0.042052146 0.11731903 -0.15506615 0.0063580726 -0.15429358 0.1511722 0.12745973 0.2576985 -0.25486213 -0.0709463 0.17983761 0.054027 -0.09884228 -0.24595179 -0.093028545 -0.028203879 0.094398156 0.09233813 0.029291354 0.13110267 0.15682974 -0.016919162 0.23927948 -0.1343307 -0.22422817 0.14634751 -0.064993896 0.4703685 -0.027190214 0.06224946 -0.091360025 0.21490277 -0.19562101 -0.10032754 -0.09056772 -0.06203493 -0.18876675 -0.10963594 -0.27734384 0.12616494 -0.02217992 -0.16058226 -0.080475815 0.026953284 0.110732645 0.014894041 0.09416802 0.14299914 -0.1594008 -0.066080004 -0.007995227 -0.11668856 -0.13081996 -0.09237365 0.14741232 0.09180138 0.081735 0.3211204 -0.0036552632 -0.047030564 -0.02311798 0.048961394 0.08669574 -0.06766279 -0.50028914 -0.048515294 0.14144728 -0.032994404 -0.11954345 -0.14929578 -0.2388355 -0.019883996 -0.15917352 -0.052084364 0.2801028 -0.0029121689 -0.054581646 -0.47385484 0.17112483 -0.12066923 -0.042173345 0.1395337 0.26115036 0.012869649 0.009291686 -0.0026459037 -0.075331464 0.017840583 -0.26869613 -0.21820338 -0.17084768 -0.1022808 -0.055290595 0.13513643 0.12362477 -0.10980586 0.13980341 -0.20233242 0.08813751 0.3849736 -0.10653763 -0.06199595 0.028849555 0.03230154 0.023856193 0.069950655 0.19310954 -0.077677034 -0.144811'
GLOVE_AVG_MISS_VEC_300 = np.array(GLOVE_AVG_MISS_STRING_300.split(" "))

# returns data which contains a list of sentences,
# option to load as list of dictionaries or tuple
# e.g. {'word': 'in', 'pos_tag': 'IN'} or ('in', 'IN')


def load_data(filename, format, infer=False):
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
                if (infer):
                    fields = line.split()
                    word = fields[0]
                    pos_tag = ""
                else:
                    word, pos_tag, _ = line.split()
                if (format == 'dictionary'):
                    sentence.append({'word': word, 'pos_tag': pos_tag})
                if (format == 'tuple'):
                    sentence.append((word, pos_tag))
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
    with open("./models/"+filename, 'wb') as file:
        pickle.dump(model, file)


def load_model(filename):
    with open("./models/"+filename, 'rb') as file:
        return pickle.load(file)

# generate files of format {embedding tag} for each word. Each scalar in vector is a separate column


def embedding(input_dir, output_dir):
    model = api.load("glove-twitter-25")
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        with open(input_path, 'r') as input_file, open(output_path, 'w') as output_file:
            for line in input_file:
                line = line.strip()
                if not line:
                    output_file.write('\n')  # write empty lines as they are
                else:
                    fields = line.split()
                    word = fields[0].lower()
                    try:
                        word = model.get_vector(word)
                    except:
                        word = model.get_vector('unk')
                    pos_tag = fields[1]
                    for v in word:
                        output_file.write(f'{v} ')
                    output_file.write(f'{pos_tag}\n')
        print(f'{file_name} vectorized.')


glob_model = api.load("glove-twitter-100")


def word2emb(word):
    try:
        word = glob_model.get_vector(word)
    except:
        word = glob_model.get_vector("unk")
    return word


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
            f'{w}_end': i == len(sentence)-1,
            f'{w}_start': i == 0,
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
        for i, v in enumerate(word2emb(word)):
            features[f'{w}_{i}'] = v
    return features

# get features for entire data


def extract_features(sentences):
    return [[feature_extractor(sentence, i) for i in range(len(sentence))] for sentence in sentences]

# get labels for each training data


def extract_labels(sentences):
    return [[token['pos_tag'] for token in sentence] for sentence in sentences]


def main():
    # clean_data('./data/', './tagonly/')
    # generate_feature_file('./tagonly/train.txt', './features/train.txt')
    # generate_feature_file('./tagonly/test.txt', './features/test.txt')
    # embedding('./tagonly/','./embedding/')
    pass


if __name__ == "__main__":
    main()
