%-----------------------------------------------------------------------%
%   generate matrix with distance constraints
%
%   Shawn Jiang
%   04/11/2016
%
%   input: n and m denote the size of V, K denotes the length of the latent
%       dimension, num1 and num2 denotes the numbers of distance
%       constraints for rows of W and columns of H
%   output: V is a n*m matrix, L1 and L2 are distance constraints
%
%   this algorithm randomly generates two matrices W and H, then obtain
%       distance constraints from the existing W and H. If a randomly pick
%       of three rows/columns satisfies the constraint format, and it does
%       not conflict with previous constraints (using map strategy), we
%       keep it, otherwise abandon it.
%-----------------------------------------------------------------------%
function [V, L1, L2] = randmx(N, M, K, num1, num2)
%   initialization
W = rand(N, K);
H = rand(K, M);
V = W * H;  %get V directly
V = (V - min(min(V))) / (max(max(V)) - min(min(V)));
L1 = genpc(W, num1);
L2 = genpc(H.', num2);

end