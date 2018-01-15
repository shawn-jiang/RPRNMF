%-------------------------------------------------------------------------%
%   derivative of symmetric divergence
%   g(x,y) = log(x/y) + (x-y)/x
%
%   Shawn Jiang
%   03/11/2016
%-------------------------------------------------------------------------%
function res = dsd(X, Y)
res = log(max(X, eps) ./ max(Y, eps)) + (X - Y) ./ max(X, eps);
end