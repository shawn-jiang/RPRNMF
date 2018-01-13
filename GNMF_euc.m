%-----------------------------------------------------------------------%
%  Graph Ragularized Nonnegative Matrix Factorization (GNMF)
%  using Euclidean Distance
%
%  Shawn Jiang
%  10/30/2016
%
%   input
%       Sw: similarity matrix for the first factorized matrix
%       Sh: similarity matrix for the second factorized matrix
%
%   We extends the original GNMF, which only has one similarity
%   matrix, to this algorithm having two similarity matrices
%-----------------------------------------------------------------------%
function [W, H, mse, csr] = GNMF_euc(V,K,Sw,Sh,L1,L2,Mask,W,H)
% if no input
if nargin == 0
    N = 50;
    M = 60;
    K = 20;
    [V, L1, L2] = randmx(N, M, K, 10, 10);
    Sw = smx(L1, N);
    Sh = smx(L2, M);
    Mask = V ~= 0;
else
    N = size(V, 1);
    M = size(V, 2);
end

% parameter
lambda = 100;

% initialization
if ~exist('Mask', 'var') || numel(Mask) == 0
    Mask = ones(N,M);   
    V(V == 0) = eps;
end

if ~exist('W','var') || ~exist('H','var')
    W = rand(N,K);
    H = rand(K,M);
end

count = 500; % iteration times
mse = zeros(count, 1); % Mean Squared Error
csr = zeros(count, 1); % Constraints Satisfied Rate

% diagnal similarity matrices
Dw = diag(sum(Sw));
Dh = diag(sum(Sh));

% if the factorising matrix is huge, use sparse version
if numel(V) > 1e8
    sp = true;
else
    sp = false;
end

if sp % if the matrix is huge, then use sparse version
    for c = 1 : count
        if mod(c, round(count/10)) == 0
            fprintf('%d ', c / round(count/10));
        end
        if c == count
            fprintf('\n');
        end
        
        for k = 1 : K
            % update W
            Temp1 = (Mask .* V) * H(k,:)';
            Temp2 = mulm(W, H, Mask) * (H(k,:).');
            if numel(Sw) == 0
                W(:,k) = W(:,k) .* Temp1 ./ max(Temp2, eps);
            else
                W(:,k) = W(:,k) .* (Temp1 + lambda * Sw * W(:,k)) ./ ...
                    max(Temp2 + lambda * Dw * W(:,k), eps);
            end

            % update H
            Temp3 = W(:,k)' * (Mask .* V);
            Temp4 = (W(:,k).') * mulm(W, H, Mask); 
            if numel(Sh) == 0
                H(k,:) = H(k,:) .* Temp3 ./ max(Temp4, eps);
            else
                H(k,:) = H(k,:) .* (Temp3 + lambda * H(k,:) * Sh) ./ ...
                    max(Temp4 + lambda * H(k,:) * Dh, eps);
            end
        end   

        % save the MSE
        mse(c,1) = calcerr(W, H, V, Mask, 'mse');

        % calculate the accuracy of generated constraints
        csr(c,1) = calccsr(W, H, L1, L2, 'mse');
        
        % print 
%         fprintf('MSE,CSR: %f,%f\n', mse(c,1), csr(c,1));  
    end
else % if the matrix is not huge, use full version
    for c = 1 : count 
        if mod(c, round(count/10)) == 0
            fprintf('%d ', c / round(count/10));
        end
        if c == count
            fprintf('\n');
        end
        
        for k = 1 : K
            % update W
            Temp1 = (Mask .* V) * H(k,:)';
            Temp2 = Mask .* (W * H) * (H(k,:).');
            if numel(Sw) == 0
                W(:,k) = W(:,k) .* Temp1 ./ max(Temp2, eps);
            else
                W(:,k) = W(:,k) .* (Temp1 + lambda * Sw * W(:,k)) ./ ...
                    max(Temp2 + lambda * Dw * W(:,k), eps);
            end

            % update H
            Temp3 = W(:,k)' * (Mask .* V);
            Temp4 = (W(:,k).') * (Mask .* (W * H));   
            if numel(Sh) == 0
                H(k,:) = H(k,:) .* Temp3 ./ max(Temp4, eps);
            else
                H(k,:) = H(k,:) .* (Temp3 + lambda * H(k,:) * Sh) ./ ...
                    max(Temp4 + lambda * H(k,:) * Dh, eps);
            end
        end   

        % save the MSE
        mse(c,1) = calcerr(W, H, V, Mask, 'mse');

        % calculate the accuracy of generated constraints
        csr(c,1) = calccsr(W, H, L1, L2, 'mse');
        
        % print 
%         fprintf('MSE,CSR: %f,%f\n', mse(c,1), csr(c,1));  
    end
end

% plot
% ecplot(mse, csr, 'mse');

% return
mse = mse(end);
csr = csr(end);

end