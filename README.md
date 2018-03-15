# AutoenCODE
AutoenCODE is a Deep Learning infrastructure that allows to *encode* source code fragments into vector representations, which can be used to learn similarities.

This repository contains code, data, and instructions on how to learn sentence-level embeddings for a given textual corpus (source code, or any other textual corpus). 
The learned embeddings (i.e., continous-valued vectors) can then be used to identify similarities among the sentences in the corpus.
AutoenCODE uses a Neural Network Language Model ([word2vec](https://arxiv.org/pdf/1301.3781.pdf)[3]), which pre-trains word embeddings in the corpus, and a *Recursive* Neural Network ([Recursive Autoencoder](http://ai.stanford.edu/~ang/papers/emnlp11-RecursiveAutoencodersSentimentDistributions.pdf)[4]) that recursively combines embeddings to learn sentence-level embeddings.

AutoenCODE was built by [Martin White](http://www.cs.wm.edu/~mgwhite/) and [Michele Tufano](http://www.cs.wm.edu/~mtufano/) and used and adapted in the context of the following research projects. If you are using AutoenCODE for research purposes, please cite:
- [1] Deep Learning Code Fragments for Code Clone Detection
- [2] Deep Learning Similarities from Different Representations of Source Code

The repository contains the original source code for [word2vec](https://arxiv.org/pdf/1301.3781.pdf)[3] and a forked/modified implementation of a [Recursive Autoencoder](http://ai.stanford.edu/~ang/papers/emnlp11-RecursiveAutoencodersSentimentDistributions.pdf)[4]. Please refer to the bibliography section to appropriately cite the following papers:

- [3] Efficient Estimation of Word Representations in Vector Space
- [4] Semi-supervised Recursive Autoencoders for Predicting Sentiment Distributions


# Corpus
With the term *corpus* we refer to a collection of sentences for which we aim to learn vector representations (embeddings).
Each *sentence* can be anything in textual format: a natural language phrase or chapter, a piece of source code (expressed as plain code or stream of lexical/AST terms), etc.

A single text file contains the entire corpus where each line represents a sentence in the corpus. 
An example can be found in `data/corpus.src`.


# Learn *Word* embeddings
In this stage we use word2vec to train a language model in order to learn word embeddings for each *term* in the corpus. These vectors will be used as pre-trained embeddings for the recursive autoencoder.
Other language models can be used to learn word embeddings, such as an RNN LM ([RNNLM Toolkit](http://www.fit.vutbr.cz/~imikolov/rnnlm/rnnlm-demo.pdf)).

### Build word2vec

The folder `bin/word2vec` contains the [source code](https://code.google.com/archive/p/word2vec/source/default/source) for [word2vec](https://arxiv.org/pdf/1301.3781.pdf). You can build the program with:
```bash
cd bin/word2vec
make
```

### run_word2vec.sh

`run_word2vec.sh` computes word embeddings for any text corpus. The inputs are: 
- the path of the directory containing the text corpus `corpus.src`;
- the path of the output directory;
- the size of the word vectors.

The output of word2vec is written into the `word2vec.out` file. The number of lines in the output is equal to the vocabulary size plus one. The first line is a header that contains the vocabulary size and the number of hidden units. Each subsequent line contains a lexical element first and then its embedding splayed on the line. For example, if the size of the word vectors is equal to 400, then the lexical element `public` will begin a line in `word2vec.out` followed by 400 doubles each separated by one space. This output serves as a dictionary that maps lexical elements to continuous-valued vectors. These vectors can be visualized using a dimensionality reduction technique such as [t-SNE](http://www.cs.toronto.edu/~hinton/absps/tsne.pdf).

### run_postprocess.py

`bin/run_postprocess.py` is a utility for parsing word2vec output. Run the script as follow: 
```
./run_postprocess.py --w2v <path/to/word2vec.out>  --src <path/to/corpus/folder/>
```
Where `<path/to/word2vec.out>` is the path to the `word2vec.out` file, and `<path/to/corpus/folder/>` is the path to the directory containing the `corpus.src` file.
The utility parses `word2vec.out` into a `vocab.txt` (containing the list of terms) and an `embed.txt` (containing the matrix of embeddings). Then the utility uses the index of each term in the list of terms to transform the src2txt `.src` files into `.int` files where the lexical elements are replaced with integers. In other words, suppose the lexical element `public` is listed on line #5 of `vocab.txt`. The embedding for `public` will be on line #5 of `embed.txt` and every instance of `public` in `corpus.src` will be replaced with the number 5 in `corpus.int`.


# Learn *Sentence* embeddings
In this stage we use a recursive autoencoder which recursively combines embeddings - starting from the word embeddings generated in the previous stage - to learn sentence-level embeddings. Then, distances among the embeddings are computed and saved in a distance matrix which can be analyzed in order to discover similarities among the sentences in the corpus.

### Recursive Autoencoder
`rae/run_rae.sh` runs the recursive autoencoder. The inputs are:
- the path of the directory containing the post-process files;
- the path of the output directory;
- the maximum sentence length used during the training (longer sentences will not be used for training);
- number of maximum training iterations.

The script invokes the matlab code `main.m`. It logs the machine name and Matlab version. Then it preprocesses the data, sets the architecture, initializes the model, trains the model, and computes/saves the similarities among the sentences. The `minFunc` log is printed to `${ODIR}/logfile.log`.

In addition to the log files, the program also saves the following files:
* `data.mat` contains the input data including the `corpus`, `vocabulary` (a 1-by-|V| cell array), and `We` (the m-by-|V| word embedding matrix where m is the size of the word vectors). So columns of `We` correspond to word embeddings.
* `corpus.dist.matrix.mat` contains the distance matrix saved as matlab file. The values in the distance matrix are doubles that represent the Euclidean distance between two sentences. In particular, the cell *(i,j)* contains the Euclidean distance between the *i*-th sentence (i.e., *i*-th line in `corpus.src`) and the *j*-th sentence in the corpus.
* `corpus.dist.matrix.csv` contains the distance matrix saved as .csv file.
* `corpus.sentence_codes.mat` contain the embeddings for each sentence in the corpus. The `sentence_codes` object contains the representations for sentences, and the pairwise Euclidean distance between these representations are used to measure similarity.
* `detector.mat` contains `opttheta` (the trained clone detector), `hparams`, and `options`.

The distance matrix can be used to sort sentences with respect to similarity in order to identify code clones.


# Execution Example 
The repository also contains input and output example data in `data/` and `out/` folders. The following lines of code perform the steps explained above and generated the output data.
```bash
cd bin/
./run_word2vec.sh ../data/ ../out/word2vec/ 100
./run_postprocess.py --w2v ../out/word2vec/word2vec.out  --src ../data/
cd rae/
./run_rae.sh ../../out/word2vec/ ../../out/rae/ 50 2
```

# Bibliography

#### [1] Deep Learning Code Fragments for Code Clone Detection

```
@inproceedings{White:2016:DLC:2970276.2970326,
 author = {White, Martin and Tufano, Michele and Vendome, Christopher and Poshyvanyk, Denys},
 title = {Deep Learning Code Fragments for Code Clone Detection},
 booktitle = {Proceedings of the 31st IEEE/ACM International Conference on Automated Software Engineering},
 series = {ASE 2016},
 year = {2016},
 isbn = {978-1-4503-3845-5},
 location = {Singapore, Singapore},
 pages = {87--98},
 numpages = {12},
 url = {http://doi.acm.org/10.1145/2970276.2970326},
 doi = {10.1145/2970276.2970326},
 acmid = {2970326},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {abstract syntax trees, code clone detection, deep learning, language models, machine learning, neural networks},
} 
```

#### [2] Deep Learning Similarities from Different Representations of Source Code

```
@inproceedings{Tufano:MSR:2018,
 author = {Tufano, Michele and Watson, Cody and Bavota, Gabriele and Di Penta, Massimiliano and White, Martin and Poshyvanyk, Denys},
 title = {Deep Learning Similarities from Different Representations of Source Code},
 booktitle = {Proceedings of the 15th International Conference on Mining Software Repositories},
 series = {MSR '18},
 year = {2018},
 isbn = {978-1-4503-5716-6/18/05},
 location = {Gothenburg, Sweden},
 url = {https://doi.org/10.1145/3196398.3196431},
 doi = {10.1145/3196398.3196431},
 publisher = {ACM},
 keywords = {deep learning, code similarities, neural networks}
}
```



#### [3] Efficient Estimation of Word Representations in Vector Space

```
@article{DBLP:journals/corr/abs-1301-3781,
  author    = {Tomas Mikolov and
               Kai Chen and
               Greg Corrado and
               Jeffrey Dean},
  title     = {Efficient Estimation of Word Representations in Vector Space},
  journal   = {CoRR},
  volume    = {abs/1301.3781},
  year      = {2013},
  url       = {http://arxiv.org/abs/1301.3781},
  archivePrefix = {arXiv},
  eprint    = {1301.3781},
  timestamp = {Wed, 07 Jun 2017 14:42:25 +0200},
  biburl    = {http://dblp.org/rec/bib/journals/corr/abs-1301-3781},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```

#### [4] Semi-supervised recursive autoencoders for predicting sentiment distributions

```
@inproceedings{Socher:2011:SRA:2145432.2145450,
 author = {Socher, Richard and Pennington, Jeffrey and Huang, Eric H. and Ng, Andrew Y. and Manning, Christopher D.},
 title = {Semi-supervised Recursive Autoencoders for Predicting Sentiment Distributions},
 booktitle = {Proceedings of the Conference on Empirical Methods in Natural Language Processing},
 series = {EMNLP '11},
 year = {2011},
 isbn = {978-1-937284-11-4},
 location = {Edinburgh, United Kingdom},
 pages = {151--161},
 numpages = {11},
 url = {http://dl.acm.org/citation.cfm?id=2145432.2145450},
 acmid = {2145450},
 publisher = {Association for Computational Linguistics},
 address = {Stroudsburg, PA, USA},
} 
```
