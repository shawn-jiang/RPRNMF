%-----------------------------------------------------------------------%
%  Repmat calculation in divergence version MF algorithms with mask matrix
%
%
%  Shawn Jiang
%  25/10/2016
%-----------------------------------------------------------------------%
function T = mrepmat(Hk, N, Mask)
% can't use T = Mask .* repmat(Hk, N, 1) because repmat(Hk, N, 1) is 
% a huge full matrix
[row,col] = find(Mask);
val = Hk(col);
T = sparse(row, col, val, N, size(Hk,2));
end