function mainPreprocess(IDIR,ODIR,MAX_SENTENCE_LENGTH,MAX_ITER)
t1 = clock;

global logger
logger = log4m.getLogger(strcat(ODIR,'/prep.log'));
logger.setCommandWindowLevel(logger.WARN);
logger.setLogLevel(logger.TRACE);

addpath(genpath('tools'));

[~,result] = system('hostname');
logger.info('main',sprintf('%s,%s',deblank(result),version));

[corpora,labels,vocabulary,We] = preprocess(IDIR,ODIR,MAX_SENTENCE_LENGTH);
