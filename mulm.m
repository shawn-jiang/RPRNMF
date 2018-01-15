%-----------------------------------------------------------------------%
%  The product of two matrices with mask matrix, the size is huge
%
%
%  Shawn Jiang
%  25/10/2016
%-----------------------------------------------------------------------%
function V = mulm(W, H, Mask)
% can't use V = Mask .* (W * H) because W * H is a huge full matrix
[row,col] = find(Mask);
V = double(Mask);
V(Mask) = dot(W(row,:), H(:,col)', 2);
end