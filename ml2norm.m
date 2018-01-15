%-----------------------------------------------------------------------%
%  Calculate the L2 norm of a matrix's rows/columns
%
%  Shawn Jiang
%  25/10/2016
%-----------------------------------------------------------------------%
function res = ml2norm(A, dim)
[N,M] = size(A);
if dim == 1
    res = zeros(1,M);
    for i = 1 : M
        res(i) = norm(A(:,i),2);
    end
elseif dim == 2
    res = zeros(N,1);
    for i = 1 : N
        res(i) = norm(A(i,:),2);
    end
end
end