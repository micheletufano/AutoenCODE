#!/bin/bash

: ${3?"usage: ${0} <src> <out> <size>"}
SRC="${1}/corpus.src"
OUT="${2}/word2vec.out"

mkdir -p $(dirname ${SRC})
mkdir -p $(dirname ${OUT})

options="-size ${3} -windows 10 -sample 1e-4 -hs 1 -negative 0 -iter 20 -min-count 1 -cbow 0"

time ./word2vec/word2vec -train ${SRC} -output ${OUT} ${options}
