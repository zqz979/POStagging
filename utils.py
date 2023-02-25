import os
import nltk
import gensim.downloader as api

def clean_data(input_dir, output_dir):
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
                    word = fields[0]
                    pos_tag = fields[1]
                    output_file.write(f'{word} {pos_tag}\n')
        print(f'{file_name} cleaned.')

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = f.read().strip()
    sents = data.split('\n\n')
    return [[nltk.tag.str2tuple(wordtag,sep=" ") for wordtag in sent.split('\n')] for sent in sents]

# generate files with features as columns
# we can modify this function for generating more meaningful features
def generate_feature_file(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    with open(output_file, "w") as f:
        for i, line in enumerate(lines):
            if line.strip():
                word, tag = line.split()
                features = []
                # Add word suffix as a feature
                features.append(word[-3:])
                # Add word prefix as a feature
                features.append(word[:3])
                # Add word length as a feature
                features.append(str(len(word)))
                # Add tag of the previous word as a feature
                if i > 0 and lines[i-1].strip():
                    prev_word, prev_tag = lines[i-1].split()
                    features.append(prev_tag)
                else: # start symbol
                    features.append('<s>')
                # Add tag of the next word as a feature
                if i < len(lines)-1 and lines[i+1].strip():
                    next_word, next_tag = lines[i+1].split()
                    features.append(next_tag)
                else: # end symbol
                    features.append('<e>')
                f.write(f"{word} {' '.join(features)} {tag}\n")
            else:
                f.write("\n")
    print(f'features generated for {input_file}.')

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


def main():
    # clean_data('./data/', './tagonly/')
    # generate_feature_file('./tagonly/train.txt', './features/train.txt')
    # generate_feature_file('./tagonly/test.txt', './features/test.txt')
    # embedding('./tagonly/','./embedding/')
    pass
if __name__ == "__main__":
    main()
    