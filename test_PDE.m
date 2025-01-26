addpath('support_fun/')
rng(0);

% example 9.1 in "A SUBSPACE-CONJUGATE GRADIENT METHOD FOR LINEAR MATRIX
% EQUATIONS" by DAVIDE PALITTA, MARTINA IANNACITO, AND VALERIA SIMONCINI
fprintf('PDE example\n')

n=8000;
% nonconstant coef  in -(a(x)u_x)_x. use a=1 to get back the laplacian
delta=-1/10;
aa=delta*exp(-linspace(0,1,2*(n+2)-1))';
e=ones(n,1);
D= spdiags([-e,e],-1:0,n+1,n);
A=-n^2*D'*spdiags(aa(2:2:2*n+2),0:0,n+1,n+1)*D;

% change this for a slower decay in the singular values of the solution
mu=sin(pi*linspace(0,1,n+2))'; fprintf('mu sin\n');
%mu=exp(pi*linspace(0,1,n+2))'; fprintf('mu exp\n');

M=spdiags(mu(2:n+1),0:0,n,n);
I=speye(n);


OP={A,I,M;I,A,M};
C={e,1,e};
normR0 = n; %sqrt(trace( (C{3}'*C{3})* (C{1}'*C{1}) )); 
a =  2 * (1 - cos(pi / (n+1))); % = eigs(T, 1, 'smallestreal');
b =  2 * (1 - cos(n * pi / (n+1))); % = eigs(T, 1, 'largestreal');
[p, q] = zolotarev_poles(8, a, b, a, b);
p=n^2*p;
q=n^2*q;

% define the parameters for sscg
maxrank=20;
tol=1e-8;
maxiter=100;
toltrunc=1e-12;
% iprec defines what type of preconditioning operator we are using
% iprec        vector of length 3
%              if iprec(1) equal to 1, select 1-term  preconditioner
%              if iprec(1) equal to 2, select 2-terms  preconditioner
%              else no preconditioner used
%              iprec(2:3) indexes of the PREC matrices to select
iprec = [2,1,2];
Prec={A,A, I, I;p,q,0,0};


%% subspacePCG det
% type_res=1, perform deterministic update of the residual (expensive in
% terms of storage allocation)
type_res=1;
% max rank to compute the residual matrix
maxrankR=length(OP)*maxrank;
tic
[xl,xc,xr]=sscg(OP, C, tol, maxiter, toltrunc,maxrank,maxrankR,Prec,iprec,type_res);
timeSUBCGdet=toc;
L=-e;
R=e;
CC = C{2};
fprintf('\nCompute true residual norm\n')
for i=1:length(OP)
    L=[L OP{1,i}*xl];
    R=[R OP{2,i}*xr];
    CC=blkdiag(CC,xc);
end
[QL,RL]=qr(L,0);
[QR,RR]=qr(R,0);
fprintf('\n\t=============================\n')
fprintf('SubspaceCG (full res) \t E-time %.4f sec, Real abs. res. %e,Real rel. res. %e\n', timeSUBCGdet, norm(RL*CC*RR','fro'),...
    norm(RL*CC*RR','fro')/normR0)
fprintf('\n\t=============================\n')


%% subspacePCG rand
% type_res=2, perform randomized update of the residual (cheaper in terms
% of storage allocation)
type_res=2;
% max rank to compute the residual matrix
maxrankR=2*maxrank;
tic
[xl,xc,xr]=sscg(OP, C, tol, maxiter, toltrunc,maxrank,maxrankR,Prec,iprec,type_res);
timeSUBCGrand=toc;
L=-e;
R=e;
CC = C{2};
fprintf('\nCompute true residual norm\n')
for i=1:length(OP)
    L=[L OP{1,i}*xl];
    R=[R OP{2,i}*xr];
    CC=blkdiag(CC,xc);
end
[QL,RL]=qr(L,0);
[QR,RR]=qr(R,0);
fprintf('\n\t=============================\n')
fprintf('SubspaceCG (rand) \t E-time %.4f sec, Real abs. res. %e, Real rel. res. %e\n', timeSUBCGrand, norm(RL*CC*RR','fro'),norm(RL*CC*RR','fro')/normR0)
fprintf('\n\t=============================\n')





