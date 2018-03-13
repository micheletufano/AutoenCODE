function computeSimilarities(ODIR)

addpath(genpath('tools'));

global logger
logger = log4m.getLogger(strcat(ODIR,'/simil.log'));
logger.setCommandWindowLevel(logger.WARN);
logger.setLogLevel(logger.TRACE);

pp = parpool(feature('numCores'));

load(strcat(ODIR,'/data.mat'));
load(strcat(ODIR,'/detector.mat'));

[hparams,options,func,func_prime] = architecture(We,1);
freq = histc(cell2mat(corpora.types'),1:hparams.vocabulary_size); % XXX </s>
freq = freq / sum(freq);

similarities(ODIR, ...
    opttheta, ...
    hparams, ...
    corpora, ...
    labels, ...
    freq, ... % FIXME
    func, ...
    func_prime, ...
    {'types', 'methods'});
