function features = getfeatures(words_reIndexed, ...
    beta, ...
    We, ...
    We2, ...
    W1,W2,W3,W4,b1,b2,b3,Wcat,bcat, ...
    alpha_cat, ...
    hidden_size, ...
    labels, ...
    freq, ...
    func, ...
    func_prime, ...
    training)

global logger

% flag
% 1 - top node only
% 2 - average of all nodes

num_examples = length(words_reIndexed);

features1 = zeros(num_examples,hidden_size);
features2 = zeros(num_examples,hidden_size);

features = zeros(num_examples,hidden_size,2);

inference_times = NaN(num_examples,1);

parfor ii = 1:num_examples
    words_rI = words_reIndexed{ii};
    nn = length(words_rI);

    freq_here = freq(words_rI);

    L = We(:,words_rI);
    if training == 1
        words_embedded = We2(:,words_rI) + L;
    else
        words_embedded = L;
    end

    t1 = clock;    
    Tree = forward([], ... % allKids
        W1,W2,W3,W4,b1,b2,b3,Wcat,bcat, ...
        alpha_cat, ...
        0, ... % updateWcat
        beta, ...
        words_embedded, ...
        0, ... % labels(:,ii), ...
        hidden_size, ...
        nn, ... % sl
        freq_here, ... % freq
        func, ...
        func_prime);
    t2 = clock;

    inference_times(ii) = etime(t2,t1);
    
    if nn > 1
        features1(ii,:) = Tree.nodeFeatures(:,end)';

        tempFeatures = zeros(2*nn-1,hidden_size);
        for i=1:2*nn-1
            tempFeatures(i,:) = Tree.nodeFeatures(:,i)';
        end
        features2(ii,:) = sum(tempFeatures)/(2*nn-1);
    elseif nn == 1
        features1(ii,:) = Tree.nodeFeatures(:,1);
        features2(ii,:) = Tree.nodeFeatures(:,1);
    else
        features1(ii,:) = zeros(hiddenSize,1);
        features2(ii,:) = zeros(hiddenSize,1);
    end
end
%{
if any(isnan(inference_times))
    logger.error('features','missing inference times');
end

Y = prctile(inference_times,[0,25,50,75,100]);
logger.info('features',sprintf('prctiles=[%.6f,%.6f,%.6f]',Y(2),Y(3),Y(4)));
logger.info('features',sprintf('mean=%.6f',mean(inference_times)));
%}
features(:,:,1) = features1;
features(:,:,2) = features2;
end
