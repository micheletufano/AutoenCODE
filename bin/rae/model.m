function [W1 W2 W3 W4 b1 b2 b3 Wcat bcat We] = model(Wcat_flag, ...
    theta, ...
    hs, ... % hparams.hidden_size
    cs, ... % hparams.category_size
    vs)     % hparams.vocabulary_size

W1 = reshape(theta(1:hs*hs),hs,hs);
W2 = reshape(theta(hs*hs*1+1:hs*hs*2),hs,hs);
W3 = reshape(theta(hs*hs*2+1:hs*hs*3),hs,hs);
W4 = reshape(theta(hs*hs*3+1:hs*hs*4),hs,hs);
b1 = theta(hs*hs*4+1:hs*hs*4+hs);
b2 = theta(hs*hs*4+hs*1+1:hs*hs*4+hs*2);
b3 = theta(hs*hs*4+hs*2+1:hs*hs*4+hs*3);

if Wcat_flag
    Wcat = reshape(theta(hs*hs*4+hs*3+1:hs*hs*4+hs*3+cs*hs),cs,hs);
    bcat = theta(hs*hs*4+hs*3+cs*hs+1:hs*hs*4+hs*3+cs*hs+cs);
    We = reshape(theta(hs*hs*4+hs*3+cs*hs+cs+1:end),hs,vs);
else
    Wcat = [];
    bcat = [];
    We = reshape(theta(hs*hs*4+hs*3+1:end),hs,vs);
end
