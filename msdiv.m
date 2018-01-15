%-----------------------------------------------------------------------%
%  Calculate the sdiv of a two matrices' rows/columns
%
%  Shawn Jiang
%  25/10/2016
%-----------------------------------------------------------------------%
function res = msdiv(A, B, dim)
if dim == 1
    res = sum((A - B) .* log(max(A,eps) ./ max(B,eps)) / 2, 1);
elseif dim == 2
    res = sum((A - B) .* log(max(A,eps) ./ max(B,eps)) / 2, 2);
end
end