function theta = initialize(hparams,We)

global logger
logger.trace('initialize','initializing model');

r  = sqrt(6) / sqrt(2 * hparams.hidden_size + 1); % U[-r, r]

W1 = rand(hparams.hidden_size,hparams.hidden_size) * 2 * r - r;
W2 = rand(hparams.hidden_size,hparams.hidden_size) * 2 * r - r;
W3 = rand(hparams.hidden_size,hparams.hidden_size) * 2 * r - r;
W4 = rand(hparams.hidden_size,hparams.hidden_size) * 2 * r - r;

b1 = zeros(hparams.hidden_size,1);
b2 = zeros(hparams.hidden_size,1);
b3 = zeros(hparams.hidden_size,1);

Wcat = rand(hparams.category_size,hparams.hidden_size) * 2 * r - r;
bcat = zeros(hparams.category_size,1);

%We = 1e-3 * (rand(hparams.hidden_size,hparams.vocabulary_size) * 2 * r - r);

theta = [W1(:);W2(:);W3(:);W4(:);b1(:);b2(:);b3(:);Wcat(:);bcat(:);We(:)];
