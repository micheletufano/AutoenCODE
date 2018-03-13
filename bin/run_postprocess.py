#!/usr/bin/python

import optparse
import os

class Word2vecOutput(object):
    def __init__(self, path):
        self._path = path
        self._vocab = {}
        with open(self._path) as fw:
            with open(self.DIR + 'vocab.txt', 'w') as fv:
                with open(self.DIR + 'embed.txt', 'w') as fe:
                    next(fw)  # Retrieve header
                    for i, item in enumerate(fw, start=1):  # MATLAB is 1-indexed
                        word, embedding = item.strip().split(' ', 1)
                        fv.write(word + '\n')
                        fe.write(embedding + '\n')
                        self._vocab[word] = i  # Map words to ints

    @property
    def DIR(self):
        return os.path.dirname(self._path) + os.sep

    @property
    def vocab(self):
        return self._vocab

class Corpus(object):
    def __init__(self, src_dir, int_dir, granularities=['corpus']):
        self._src_dir = src_dir
        self._int_dir = int_dir
        self._granularities = granularities

    def transform(self, vocab):
        for granularity in self._granularities:
            src_path = self._src_dir + os.sep + granularity + '.src'
            int_path = self._int_dir + os.sep + granularity + '.int'
            with open(src_path) as fi:
                with open(int_path, 'w') as fo:
                    for line in fi:
                        words = line.strip().split()
                        fo.write(' '.join(str(vocab[w]) for w in words) + '\n')

if __name__ == '__main__':
    # Build *.int
    parser = optparse.OptionParser()
    parser.add_option('--w2v', help='/path/to/word2vec.out')
    parser.add_option('--src', help='/path/to/*.src')
    (options, args) = parser.parse_args()

    word2vec_output = Word2vecOutput(options.w2v)

    corpus = Corpus(options.src, word2vec_output.DIR)
    corpus.transform(word2vec_output.vocab)

