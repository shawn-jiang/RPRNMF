%-------------------------------------------------------------------------%
%   Pairwise Relationship Constrained NMF using Euclidean Distance 
%
%   Shawn Jiang
%   03/11/2016
%-------------------------------------------------------------------------%
function [W, H, mse, csr] = RPRNMF_euc(V,K,L1,L2,lambda,Mask,W,H)
% if no input
if nargin == 0
    N = 50;
    M = 60;
    K = 20;
    lambda = 10;
    [V, L1, L2] = randmx(N, M, K, 10, 10);
    V(rand(size(V))<0.5) = 0;
    V = V * 5;
    Mask = V ~= 0;
else
    N = size(V, 1);
    M = size(V, 2);
end

maxv = max(max(V));
minv = min(min(V));
V = V / (maxv - minv);

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
ofv = zeros(count, 1); % Objective Function Value
nl1 = size(L1,1);
nl2 = size(L2,1);

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
            Ca1 = zeros(N,1);
            Ca2 = zeros(N,1);
            Da1 = ml2norm(W(L1(:,1),:) - W(L1(:,2),:), 2).^2;
            Da2 = ml2norm(W(L1(:,1),:) - W(L1(:,3),:), 2).^2;     
            for l = 1 : nl1
                q = L1(l,1);
                r = L1(l,2);
                s = L1(l,3);
                Ca1(q) = Ca1(q) + exp(Da1(l)) * W(q,k) + exp(-Da2(l)) * W(s,k);
                Ca1(r) = Ca1(r) + exp(Da1(l)) * W(r,k);
                Ca1(s) = Ca1(s) + exp(-Da2(l)) * W(q,k);
                Ca2(q) = Ca2(q) + exp(Da1(l)) * W(r,k) + exp(-Da2(l)) * W(q,k);
                Ca2(r) = Ca2(r) + exp(Da1(l)) * W(q,k);
                Ca2(s) = Ca2(s) + exp(-Da2(l)) * W(s,k);
            end
            W(:,k) = W(:,k) .* (Mask .* V * H(k,:)' + lambda/max(nl1,eps) * Ca2) ./ ...
                max(mulm(W,H,Mask) * H(k,:)' + lambda/max(nl1,eps) * Ca1, eps);

            % update H
            Cb1 = zeros(1,M);
            Cb2 = zeros(1,M);
            Db1 = ml2norm(H(:,L2(:,1)) - H(:,L2(:,2)), 1).^2;
            Db2 = ml2norm(H(:,L2(:,1)) - H(:,L2(:,3)), 1).^2;
            for l = 1 : nl2
                q = L2(l,1);
                r = L2(l,2);
                s = L2(l,3);
                Cb1(q) = Cb1(q) + exp(Db1(l)) * H(k,q) + exp(-Db2(l)) * H(k,s);
                Cb1(r) = Cb1(r) + exp(Db1(l)) * H(k,r);
                Cb1(s) = Cb1(s) + exp(-Db2(l)) * H(k,q);
                Cb2(q) = Cb2(q) + exp(Db1(l)) * H(k,r) + exp(-Db2(l)) * H(k,q);
                Cb2(r) = Cb2(r) + exp(Db1(l)) * H(k,q);
                Cb2(s) = Cb2(s) + exp(-Db2(l)) * H(k,s);
            end
            H(k,:) = H(k,:) .* (W(:,k)' * (Mask .* V) + lambda/max(nl2,eps) * Cb2) ./ ...
                max(W(:,k)' * mulm(W,H,Mask) + lambda/max(nl2,eps) * Cb1, eps);
        end
        
        % save the MSE
        mse(c,1) = calcerr(W, H, V, Mask, 'mse');
        
        % calculate the accuracy of generated constraints
        csr(c,1) = calccsr(W, H, L1, L2, 'mse');
        
        pw = lambda/max(nl1,eps) * sum(...
            exp(ml2norm(W(L1(:,1),:) - W(L1(:,2),:), 2).^2) ...
            + exp(- ml2norm(W(L1(:,1),:) - W(L1(:,3),:), 2).^2));
        ph = lambda/max(nl2,eps) * sum(...
            exp(ml2norm(H(:,L2(:,1)) - H(:,L2(:,2)), 1).^2) ...
            + exp(- ml2norm(H(:,L2(:,1)) - H(:,L2(:,3)), 1).^2));
        ofv(c,1) = log(mse(c,1) * nnz(V) + pw + ph);

        % print 
%         fprintf('%d,MSE,CSR,OFV: %f,%f,%g\n', c, mse(c,1), csr(c,1), ofv(c,1));
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
            Ca1 = zeros(N,1);
            Ca2 = zeros(N,1);
            Da1 = ml2norm(W(L1(:,1),:) - W(L1(:,2),:), 2).^2;
            Da2 = ml2norm(W(L1(:,1),:) - W(L1(:,3),:), 2).^2;     
            for l = 1 : nl1
                q = L1(l,1);
                r = L1(l,2);
                s = L1(l,3);
                Ca1(q) = Ca1(q) + exp(Da1(l)) * W(q,k) + exp(-Da2(l)) * W(s,k);
                Ca1(r) = Ca1(r) + exp(Da1(l)) * W(r,k);
                Ca1(s) = Ca1(s) + exp(-Da2(l)) * W(q,k);
                Ca2(q) = Ca2(q) + exp(Da1(l)) * W(r,k) + exp(-Da2(l)) * W(q,k);
                Ca2(r) = Ca2(r) + exp(Da1(l)) * W(q,k);
                Ca2(s) = Ca2(s) + exp(-Da2(l)) * W(s,k);
            end
            W(:,k) = W(:,k) .* (Mask .* V * H(k,:)' + lambda/max(nl1,eps) * Ca2) ./ ...
                max(Mask .* (W * H) * H(k,:)' + lambda/max(nl1,eps) * Ca1, eps);

            % update H
            Cb1 = zeros(1,M);
            Cb2 = zeros(1,M);
            Db1 = ml2norm(H(:,L2(:,1)) - H(:,L2(:,2)), 1).^2;
            Db2 = ml2norm(H(:,L2(:,1)) - H(:,L2(:,3)), 1).^2;
            for l = 1 : nl2
                q = L2(l,1);
                r = L2(l,2);
                s = L2(l,3);
                Cb1(q) = Cb1(q) + exp(Db1(l)) * H(k,q) + exp(-Db2(l)) * H(k,s);
                Cb1(r) = Cb1(r) + exp(Db1(l)) * H(k,r);
                Cb1(s) = Cb1(s) + exp(-Db2(l)) * H(k,q);
                Cb2(q) = Cb2(q) + exp(Db1(l)) * H(k,r) + exp(-Db2(l)) * H(k,q);
                Cb2(r) = Cb2(r) + exp(Db1(l)) * H(k,q);
                Cb2(s) = Cb2(s) + exp(-Db2(l)) * H(k,s);
            end
            H(k,:) = H(k,:) .* (W(:,k)' * (Mask .* V) + lambda/max(nl2,eps) * Cb2) ./ ...
                max(W(:,k)' * (Mask .* (W * H)) + lambda/max(nl2,eps) * Cb1, eps);
        end
        
        % save the MSE
        mse(c,1) = calcerr(W, H, V, Mask, 'mse');
        
        % calculate the accuracy of generated constraints
        csr(c,1) = calccsr(W, H, L1, L2, 'mse');
        
        pw = lambda/max(nl1,eps) * sum(...
            exp(ml2norm(W(L1(:,1),:) - W(L1(:,2),:), 2).^2) ...
            + exp(- ml2norm(W(L1(:,1),:) - W(L1(:,3),:), 2).^2));
        ph = lambda/max(nl2,eps) * sum(...
            exp(ml2norm(H(:,L2(:,1)) - H(:,L2(:,2)), 1).^2) ...
            + exp(- ml2norm(H(:,L2(:,1)) - H(:,L2(:,3)), 1).^2));
        ofv(c,1) = log(mse(c,1) * nnz(V) + pw + ph);

        % print 
%         fprintf('%d,MSE,CSR,OFV: %f,%f,%g\n', c, mse(c,1), csr(c,1), ofv(c,1));
    end
end

% plot
% ecplot(mse, csr, 'mse');

% return
V = V * (maxv - minv);
W = full(W * (maxv - minv));
mse = calcerr(W, H, V, Mask, 'mse');
csr = csr(end);

end