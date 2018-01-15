%-----------------------------------------------------------------------%
%  Matrix elementwise division with mask matrix
%
%
%  Shawn Jiang
%  25/10/2016
%-----------------------------------------------------------------------%
function D = mdivide(A, B, Mask)
idx = find(Mask);
D = double(Mask);
D(idx) = A(idx) ./ B(idx);
end