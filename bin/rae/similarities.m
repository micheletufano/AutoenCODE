function similarities(ODIR, ...
    opttheta, ...
    hparams, ....
    corpora, ...
    labels, ...
    freq, ...
    func, ...
    func_prime, ...
    granularities)

global logger

[W1,W2,W3,W4,b1,b2,b3,Wcat,bcat,We] = model(1, ...
    opttheta, ...
    hparams.hidden_size, ...
    hparams.category_size, ...
    hparams.vocabulary_size);

for granularity = granularities
    switch granularity{1}
        case 'corpus'
            corpus = corpora.corpus;
        otherwise
            logger.error('similarities','invalid expression');
            return
    end

    logger.trace('similarities',sprintf('encoding %s',granularity{1}));
    t1 = clock;
    features = getfeatures(corpus, ...
        0, ... % beta
        We, ... % FIXME We
        We, ... % FIXME We2
        W1,W2,W3,W4,b1,b2,b3,Wcat,bcat, ...
        hparams.alpha_cat, ...
        hparams.hidden_size, ...
        labels, ...
        freq, ...
        func, ...
        func_prime, ...
        0); % params.trainModel
    t2 = clock;
    logger.info('similarities',sprintf('encoding %s took %.0f seconds',granularity{1},etime(t2,t1)));


    %Computing distance matrix
    logger.trace('similarities',sprintf('computing %s similarities',granularity{1}));
    t1 = clock;
    sentence_codes = features(:,:,1); % num_documents-by-hparams.hidden_size
    D = pdist(sentence_codes,'euclidean');
    Z = squareform(D);
    t2 = clock;
    logger.info('similarities',sprintf('computing %s similarities took %.0f seconds',granularity{1},etime(t2,t1)));


	%Saving sentences and distance matrix
    logger.trace('similarities',sprintf('saving %s sentence_codes',granularity{1}));
    t1 = clock;
    save([strcat(ODIR,'/',granularity{1},'.dist.matrix.mat')],'Z','-v7.3');
    save([strcat(ODIR,'/',granularity{1},'.sentence_codes.mat')],'sentence_codes','-v7.3');
    t2 = clock;
    logger.trace('similarities',sprintf('saving %s sentence_codes took %.0f seconds',granularity{1},etime(t2,t1)));


	%Saving distance matrix as csv
    logger.trace('similarities',sprintf('saving %s distances',granularity{1}));
    t1 = clock;
    dlmwrite(sprintf('%s/%s.dist.matrix.csv',ODIR,granularity{1}),Z);
    t2 = clock;
    logger.trace('similarities',sprintf('saving %s distances took %.0f seconds',granularity{1},etime(t2,t1)));
end
