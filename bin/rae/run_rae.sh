#!/bin/bash

: ${4?"usage: ${0} <IDIR> <ODIR> <MAX_SENTENCE_LENGTH> <MAX_ITER>"}

IDIR=${1}
ODIR=${2}
MAX_SENTENCE_LENGTH=${3}
MAX_ITER=${4}

matlab -nodisplay -r "main('${IDIR}','${ODIR}',${MAX_SENTENCE_LENGTH},${MAX_ITER},0,'');exit"
