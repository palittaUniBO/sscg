function [x,k]=pcg_inner(Op,b,x,maxit,tol,Prec1,Prec2,iprec)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% -------------preconidtioned CG inner solver------------------------------
%
% [x,k]=pcg1_stoch_inner(Op,b,x,maxit,tol,Prec1,flagBioli,Prec2,iprec,PrecL1,PrecL2)
%
% Solve the 'projected' multiterm matrix equation matrix equation:
%
%         Pl(A1 X B1 + A2 X B2 + .... + Ap X Bp)Pr' = Pl(rhs)Pr'.
%
% using vectorization.
%
% INPUTS:
% Op           set of (2, p) matrices s.t. Op{1, j} = Pl Aj Pr' 
%              and Op{2, j} = Pl Bj Pr ' 
% b            rhs vector, such that b = vectorization(Pl(rhs)Pr')
% x            initial guess vector 
% maxit        max number of pcg iterations
% tol          stopping tolerance. The algorithm terminates if the relative
%              relative residual is lower than tol
% Prec1        left preconditioning matrix for 1-term preconditioner
%              structure, left preconditioning matrix for 2-term preconditioner
% Prec2        right preconditioning matrix for 1-term preconditioner
%              structure, right preconditioning matrix for 2-term preconditioner
% iprec        integer
%              if iprec(1) equal to 1, select 1-term  preconditioner
%              if iprec(1) equal to 2, select 2-terms  preconditioner
%              else no preconditioner used
%
% OUTPUTS:
% x            X = reshape(x, nA, nB) final approximate solution
% k            index of the last iteration performed
%
% -------------------------------------------------------------------------
% REFERENCE: A subspace-conjugate gradient method for linear matrix
% equations, D. Palitta, M. Iannacito, V. Simonici. Preprint at
% https://arxiv.org/abs/2501.02938 (2025)
% -------------------------------------------------------------------------
%
% Copyright (c): D. Palitta, M. Iannacito, V. Simonici, th January 2025.
%
% ------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = length(b);                  % inner problem-vectorized size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nh = size(Op{1,1},1);            % inner problem size
XX = reshape(x,nh,nh);           % vectorized initial guess
pp = size(Op,2);                 % number of operator terms

% Compute vec(\sum_{k=1}^{pp} A_k XX B_k)
SUM = 0;
for i = 1:pp
    SUM = SUM + Op{1,i}*XX*Op{2,i}';
end
ax = reshape(SUM,n,1);

% Compute residual associated with initial guess
r = b-ax; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

res0 = norm(b);
res = res0;                     % residual norm
mem = 1;                        % used memory
k = 0;                          % iteration idex

% start pcg iterations
while (res/res0 > tol && k<maxit)
    
    % Apply preconditioner in matrix form
    wrk=reshape(r,nh,nh);
    if iprec==2                 % 2-term preconditioner
        z = reshape(Prec1.VP*((Prec1.VP'*wrk*Prec2.VP)./Prec1.L)*Prec2.VP', ...
            nh*nh,1);
    
    elseif iprec==1             % 1-term preconditioner
        z = reshape(Prec1\wrk/Prec2,nh*nh,1);
    
    else
        z = wrk(:);
    end % end preconditioner application

    k = k+1;
    gamma = r.'*z;
    
    % Update direction vector 
    if k==1
        p=z;
    else
        beta = gamma/gamma0;
        p=z+beta*p;
    end

    % Compute vec(\sum_{k=1}^{pp} A_k P B_k)
    wrk=reshape(p,nh,nh);

    AP = 0;
    for i=1:pp
        AP = AP + Op{1,i}*wrk*Op{2,i}';
    end

    ap = reshape(AP,n,1);

    delta=p.'*ap;
    alfa = gamma/delta;
    
    % Update iterative solution
    x = x + alfa*p;
    % Update residual
    r = r - alfa*ap;
    gamma0 = gamma;
    res = norm(r);
end % end of while of pcg


