import os
import nltk
import gensim.downloader as api

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
                # last column is ignore
                word, pos_tag, _ = line.split()
                if (format == 'dictionary'):
                    sentence.append({'word': word, 'pos_tag': pos_tag})
                if (format == 'tuple'):
                    sentence.append((word,pos_tag))
        if sentence:
            data.append(sentence)
    return data

# generate files of format {embedding tag} for each word. Each scalar in vector is a separate column
def embedding(input_dir,output_dir):
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

# Get features for each sentence
def feature_extractor(sentence, i):
    word = sentence[i]['word']
    features = {
        # word itself
        'word': word,
        # suffixes
        'suffix1': word[-3:],
        'suffix2': word[-2:],
        'suffix3': word[-1:],
        # prefixes
        'prefix1': word[:3],
        'prefix2': word[:2],
        'prefix3': word[:1],
        # All-cap word can be proper nouns or abbreviations
        'word.isupper()': word.isupper(),
        # Capitalized words can be beginning of sentences, proper nouns, and acronyms.
        'word.istitle()': word.istitle(),
        # if the word is a number
        'word.isdigit()': word.isdigit(),
        # tag of previous word
        'prev_tag': '<s>' if i == 0 else sentence[i-1]['pos_tag'],
        # tag of next word
        'next_tag': '<e>' if i == len(sentence)-1 else sentence[i+1]['pos_tag'],
    }
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
    