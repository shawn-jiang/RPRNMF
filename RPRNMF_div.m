%-------------------------------------------------------------------------%
%   Pairwise Relationship Constrained NMF using Divergence
%
%   dsd(x,y) is the g(x,y) in the paper. It calculates the derivative of
%   the symmetric divergence.
%
%   Shawn Jiang
%   03/11/2016
%-------------------------------------------------------------------------%
function [W, H, md, csr] = RPRNMF_div(V,K,L1,L2,lambda,Mask,W,H)
% if no input
if nargin == 0
    N = 100;
    M = 100;
    K = 20;
    lambda = 0.2;
    [V, L1, L2] = randmx(N, M, K, 10, 10);
    V(rand(size(V))<0.5) = 0;
    Mask = V ~= 0;
else
    N = size(V, 1);
    M = size(V, 2);
end

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
md = zeros(count, 1); % Mean Divergence
csr = zeros(count, 1); % Constraints Satisfied Rate
ofv = zeros(count, 1); % Objective Function Value

% if the factorising matrix is huge, use sparse version
if numel(V) > 1e8
    sp = true;
else
    sp = false;
end

c = 1;
if ~sp % if the matrix is huge, then use sparse version
    while(c <= count)
        if mod(c, round(count/10)) == 0
            fprintf('%d ', c / round(count/10));
        end
        if c == count
            fprintf('\n');
        end
        
        W0 = W;
        H0 = H;
        for k = 1 : K
            % update W
            Ca = zeros(N,1);
            idxa = find(msdiv(W(L1(:,1),:), W(L1(:,2),:), 2) >= msdiv(W(L1(:,1),:), W(L1(:,3),:), 2));
            for l = idxa'
                q = L1(l,1);
                r = L1(l,2);
                s = L1(l,3);
                Ca(q) = Ca(q) + dsd(W(q,k), W(r,k)) - dsd(W(q,k), W(s,k));
                Ca(r) = Ca(r) + dsd(W(r,k), W(q,k));
                Ca(s) = Ca(s) - dsd(W(s,k), W(q,k));
            end
            Temp1 = sum(mdivide(V, mulm(W, H, Mask), Mask) .* mrepmat(H(k,:), N, Mask), 2);
            Temp2 = Mask * H(k,:)';
            Temp3 = Temp2 + 1/2 * lambda * Ca;
            Temp3(Temp3 < 0) = Temp2(Temp3 < 0);
            W(:,k) = W(:,k) .* Temp1 ./ max(Temp3, eps);

            % update H
            Cb = zeros(1,M);
            idxb = find(msdiv(H(:,L2(:,1)), H(:,L2(:,2)), 1) >= msdiv(H(:,L2(:,1)), H(:,L2(:,3)), 1));
            for l = idxb
                q = L2(l,1);
                r = L2(l,2);
                s = L2(l,3);
                Cb(q) = Cb(q) + dsd(H(k,q), H(k,r)) - dsd(H(k,q), H(k,s));
                Cb(r) = Cb(r) + dsd(H(k,r), H(k,q));
                Cb(s) = Cb(s) - dsd(H(k,s), H(k,q));
            end
            Temp4 = sum(mdivide(V, mulm(W, H, Mask), Mask) .* mrepmat(W(:,k)', M, Mask')', 1);
            Temp5 = W(:,k)' * Mask;
            Temp6 = Temp5 + 1/2 * lambda * Cb;
            Temp6(Temp6 < 0) = Temp5(Temp6 < 0);
            H(k,:) = H(k,:) .* Temp4 ./ max(Temp6, eps);
        end

        % penalties
        pw = msdiv(W(L1(:,1),:), W(L1(:,2),:), 2) - msdiv(W(L1(:,1),:), W(L1(:,3),:), 2);
        pw = lambda * sum(pw(pw > 0));
        ph = msdiv(H(:,L2(:,1)), H(:,L2(:,2)), 1) - msdiv(H(:,L2(:,1)), H(:,L2(:,3)), 1);
        ph = lambda * sum(ph(ph > 0));
        
        % save the MD
        md(c,1) = calcerr(W, H, V, Mask, 'md');

        % calculate the accuracy of generated constraints
        csr(c,1) = calccsr(W, H, L1, L2, 'md');

        % ofv
        ofv(c,1) = md(c,1) * nnz(Mask) + pw + ph;

        % print 
%         fprintf('%d,MD,CSR,OFV,lambda: %f,%f,%g,%f\n', c, md(c,1), csr(c,1), ofv(c,1), lambda);
        
        if c > 1 && ofv(c) > ofv(c - 1)
            W = W0;
            H = H0;
            lambda = 0.5 * lambda;
        else           
            lambda = 1.01 * lambda;
            c = c + 1;
        end
        
        if isinf(ofv(1)) || lambda < 1e-6
            error('Awful initial value.');
        end
    end
else % if the matrix is not huge, use full version
    while(c <= count)
        if mod(c, round(count/10)) == 0
            fprintf('%d ', c / round(count/10));
        end
        if c == count
            fprintf('\n');
        end
        
        W0 = W;
        H0 = H;
        for k = 1 : K
            % update W
            Ca = zeros(N,1);
            idxa = find(msdiv(W(L1(:,1),:), W(L1(:,2),:), 2) >= msdiv(W(L1(:,1),:), W(L1(:,3),:), 2));
            for l = idxa'
                q = L1(l,1);
                r = L1(l,2);
                s = L1(l,3);
                Ca(q) = Ca(q) + dsd(W(q,k), W(r,k)) - dsd(W(q,k), W(s,k));
                Ca(r) = Ca(r) + dsd(W(r,k), W(q,k));
                Ca(s) = Ca(s) - dsd(W(s,k), W(q,k));
            end
            Temp1 = sum(Mask .* V .* repmat(H(k,:), N, 1) ./ (W * H), 2);
            Temp2 = Mask * H(k,:)';
            Temp3 = Temp2 + 1/2 * lambda * Ca;
            Temp3(Temp3 < 0) = Temp2(Temp3 < 0);
            W(:,k) = W(:,k) .* Temp1 ./ max(Temp3, eps);

            % update H
            Cb = zeros(1,M);
            idxb = find(msdiv(H(:,L2(:,1)), H(:,L2(:,2)), 1) >= msdiv(H(:,L2(:,1)), H(:,L2(:,3)), 1));
            for l = idxb
                q = L2(l,1);
                r = L2(l,2);
                s = L2(l,3);
                Cb(q) = Cb(q) + dsd(H(k,q), H(k,r)) - dsd(H(k,q), H(k,s));
                Cb(r) = Cb(r) + dsd(H(k,r), H(k,q));
                Cb(s) = Cb(s) - dsd(H(k,s), H(k,q));
            end
            Temp4 = sum(Mask .* V .* repmat(W(:,k), 1, M) ./ (W * H), 1);
            Temp5 = W(:,k)' * Mask;
            Temp6 = Temp5 + 1/2 * lambda * Cb;
            Temp6(Temp6 < 0) = Temp5(Temp6 < 0);
            H(k,:) = H(k,:) .* Temp4 ./ max(Temp6, eps);
        end

        % penalties
        pw = msdiv(W(L1(:,1),:), W(L1(:,2),:), 2) - msdiv(W(L1(:,1),:), W(L1(:,3),:), 2);
        pw = lambda * sum(pw(pw > 0));
        ph = msdiv(H(:,L2(:,1)), H(:,L2(:,2)), 1) - msdiv(H(:,L2(:,1)), H(:,L2(:,3)), 1);
        ph = lambda * sum(ph(ph > 0));
        
        % save the MD
        md(c,1) = calcerr(W, H, V, Mask, 'md');

        % calculate the accuracy of generated constraints
        csr(c,1) = calccsr(W, H, L1, L2, 'md');

        % ofv
        ofv(c,1) = md(c,1) * nnz(Mask) + pw + ph;
        
        % print
%         fprintf('%d,MD,CSR,OFV,lambda: %f,%f,%g,%f\n', c, md(c,1), csr(c,1), ofv(c,1), lambda);
        
        if c > 1 && ofv(c) > ofv(c - 1)
            W = W0;
            H = H0;
            lambda = 0.5 * lambda;
        else
            lambda = 1.01 * lambda;
            c = c + 1;
        end
        
        if isinf(ofv(1)) || lambda < 1e-6
            error('Awful initial value.');
        end
    end
end

% plot
% ecplot(md, csr, 'md');

% return
md = md(end);
csr = csr(end);

end