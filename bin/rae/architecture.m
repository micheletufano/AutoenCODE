function [hparams,options,func,func_prime] = architecture(We,MAX_ITER)

global logger
logger.trace('architecture','setting architecture');

% set hyperparameters
[hparams.hidden_size,hparams.vocabulary_size] = size(We);
hparams.category_size = 1;
% relative weighting of reconstruction error and categorization error
hparams.alpha_cat = 1.0; % pure unsupervised learning
% regularization: lambda = [lambdaW,lambdaL,lambdaCat,lambdaLRAE];
hparams.lambda = [1e-05,0.0001,1e-07,0.01];
% weight of classifier cost on nonterminals
hparams.beta = 0.5; % ???
logger.info('architecture',sprintf('hiddens=%d,vocab=%d',hparams.hidden_size,hparams.vocabulary_size));

% set options for the optimizer
options.Method = 'lbfgs';
options.display = 'on';
options.maxIter = MAX_ITER;
logger.info('architecture',sprintf('maxIter=%d',options.maxIter));

% set activation function
func = @norm1tanh;
func_prime = @norm1tanh_prime;
