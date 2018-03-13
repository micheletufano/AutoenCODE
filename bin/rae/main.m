function main(IDIR,ODIR,MAX_SENTENCE_LENGTH,MAX_ITER,RESTART,CKPT_FILE)
t1 = clock;

%Set up log file
global logger
logger = log4m.getLogger(strcat(ODIR,'/log4m',num2str(RESTART),'.log'));
logger.setCommandWindowLevel(logger.WARN);
logger.setLogLevel(logger.TRACE);

addpath(genpath('tools'));
[~,result] = system('hostname');
logger.info('main',sprintf('%s,%s',deblank(result),version));

% check wheter to continue the training (restart) or start from scratch the training process
if RESTART
    load(strcat(ODIR,'/data.mat')); % Contains corpora, labels, vocabulary, We
else
    [corpora,labels,vocabulary,We] = preprocess(IDIR,ODIR,MAX_SENTENCE_LENGTH);
end


[hparams,options,func,func_prime] = architecture(We,MAX_ITER);


if RESTART
    % Checkpoint has filename convention detector.<datestr(clock)>.mat
    load(strcat(ODIR,'/',CKPT_FILE)); % Contains theta
else
    theta = initialize(hparams,We);
end


delete(gcp('nocreate'));
pp = parpool(feature('numCores'));

[opttheta,cost,freq] = train(ODIR, ...
    hparams, ...
    We, ...
    corpora.train, ...
    labels, ...
    func, ...
    func_prime, ...
    theta, ...
    options);

similarities(ODIR, ...
    opttheta, ...
    hparams, ...
    corpora, ...
    labels, ...
    freq, ... % FIXME
    func, ...
    func_prime, ...
    {'corpus'});

    
delete(gcp('nocreate'));

t2 = clock;
logger.info('main',sprintf('main.m took %.0f seconds',etime(t2,t1)));
