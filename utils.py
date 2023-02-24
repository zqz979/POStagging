import os
import nltk

def clean_data(input_path="./data/test.txt",output_path="./tagonly/test.txt"):
    with open(input_path, 'r') as input_file, open(output_path, 'w+') as output_file:
        for line in input_file:
            line = line.strip()
            if not line:
                output_file.write('\n')  # write empty lines as they are
            else:
                fields = line.split()
                word = fields[0]
                pos_tag = fields[1]
                output_file.write(f'{word} {pos_tag}\n')

def load_data(file_path="./tagonly/train.txt"):
    with open(file_path, 'r') as f:
        data = f.read().strip()
    sents = data.split('\n\n')
    return [[nltk.tag.str2tuple(wordtag,sep=" ") for wordtag in sent.split('\n')] for sent in sents]