addpath('support_fun/')
rng(0);

%sym example from Breiten-Benner (used in Shank-Simoncini-Szyld)
fprintf('Control example from Breiten-Benner\n')

NN = 320; 
maxrank = 50;
fprintf('\tNN = %d; maxrank = %d\n', NN, maxrank)
% constrcut the data
[A0,N0,B,C1] = Heat_Transfer_Model(NN,'c','a','a','a',0.9);
A=-A0; N=-N0;
I=speye(size(A));

% define the operator and the rhs
OP={A,I,-N;I,A,N};
C={B,1,B};
cl = C{1}*C{2}; cr = C{3};
normR0 = sqrt(trace((C{3}'*C{3})*(C{1}'*C{1})));

% compute poles for applying the 2-term preconditioner by ADI
n=sqrt(size(A,1));
a =  2 * (1 - cos(pi / (n+1))); % = eigs(T, 1, 'smallestreal');
b =  2 * (1 - cos(n * pi / (n+1))); % = eigs(T, 1, 'largestreal');
[p, q] = zolotarev_poles(8, a, b, a, b);
p=p*n^2;
q=q*n^2;

% define the parameters for sscg
tol=1e-6;
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
L=-cl;
R=cr;
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
fprintf(['SubspaceCG (full res) \t E-time %.4f (%d) sec, Real abs. res. %e, ' ...
    'Real rel. res. %e'], timeSUBCGdet, norm(RL*CC*RR','fro'),...
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
L=-cl;
R=cr;
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
fprintf(['SubspaceCG (rand) \t E-time %.4f (%d) sec, Real abs. res. %e, ' ...
    'Real rel. res. %e\n'], timeSUBCGrand, norm(RL*CC*RR','fro'),...
    norm(RL*CC*RR','fro')/normR0)
fprintf('\n\t=============================\n')





