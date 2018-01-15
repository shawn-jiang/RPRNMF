%-------------------------------------------------------------------------%
%   Calculate the error of a matrix factorization
%
%   Shawn Jiang
%   13/01/2017
%-------------------------------------------------------------------------%
function err = calcerr(W, H, V, Mask, type)
if numel(Mask) == 0
    num = numel(V);
else
    num = nnz(Mask);
end

try
    temp = Mask .* (W * H);
catch e
    if ~isempty(e)
        temp = mulm(W, H, Mask);
    end
end
    
if strcmp(type, 'mse') % mean square error
    err = norm(Mask .* V - temp, 'fro') ^ 2 / num;
elseif strcmp(type, 'md') % mean divergence
    err = div(Mask .* V, temp) / num;
else
    error('no such error calculation.');
end
err = full(err);
end