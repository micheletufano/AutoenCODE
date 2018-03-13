function [corpora,labels,vocabulary,We] = preprocess(IDIR,ODIR,MAX_SENTENCE_LENGTH)
t1 = clock;

global logger
logger.trace('preprocess','preprocessing data');

corpora.train = transform(IDIR,ODIR,MAX_SENTENCE_LENGTH,'corpus', true);
corpora.corpus = transform(IDIR,ODIR,MAX_SENTENCE_LENGTH,'corpus', false);

labels = zeros(1,size(corpora.train,1));

vocabulary = textscan(fopen(strcat(IDIR,'/vocab.txt')), '%s');
vocabulary = vocabulary{1}'; % 1-by-size(vocabulary,1)

We = dlmread(strcat(IDIR,'/embed.txt'));
We = We'; % m-by-size(vocabulary,1)

save([strcat(ODIR,'/data.mat')],'corpora','labels','vocabulary','We');

t2 = clock;
logger.info('preprocess',sprintf('preprocess.m took %.0f seconds',etime(t2,t1)));


function corpus = transform(IDIR,ODIR,MAX_SENTENCE_LENGTH,granularity, training)

global logger

filename = strcat(IDIR,'/',granularity,'.int');
[~,result] = system(strcat('wc -l < ',filename));
numlines = str2num(result);

corpus = cell(numlines,1);
fid = fopen(filename);
i = 1;
sentence = fgetl(fid);
while ischar(sentence)
    corpus{i} = str2double(strsplit(sentence,' '));
    sentence = fgetl(fid);
    i = i + 1;
end


if training
    logger.info('transform',sprintf('corpus has %d sentences',size(corpus,1)));
    logger.info('transform',sprintf('max sentence length=%d',MAX_SENTENCE_LENGTH));
    sizes = cellfun(@(x) size(x,2),corpus);
    corpus = corpus(sizes < MAX_SENTENCE_LENGTH);
    logger.info('transform',sprintf('training corpus has %d sentences',size(corpus,1)));
    logger.info('transform',sprintf('longest sentence in training corpus has %d terms',max(cellfun(@(x) size(x,2),corpus))));
end
