
addpath('support_fun/')

% Example from package by Powell-Simoncini-Silvester
load ./TP_five.mat; p=length(G);
rng(0);

psort=symamd(K{1});
for ind=1:p
    K1{ind}=K{ind}(psort,psort); %former K
end
fnew=fnew(psort);

Amean=K1{1};

n=size(K1{1},1);
m=size(G{1},1);
fprintf('Dimensions of the target solution: %d %d\n',n,m)
fprintf('\n')
I=speye(n);        Im=speye(m);

% define the linear operator for subspace cg:
% the matrices on the first row of OP are the ones on the left of the unkown,
% given in order. Similarly, the matrices on the second row of OP are the ones
% one the right of the unkown. Therefore, we are solving
% OP{1,1}*X*OP{2,1}+...+OP{1,p}*X*OP{2,p} = C{1}*C{2}*C{3}'
for j=1:p, OP{1,j}=K1{j};OP{2,j}=G{j};end


clear K G GG K1 
% define the rhs
C = {fnew(1:n),1,eye(m,1)};
cl = C{1}; cr = C{3};
tmp1 = cl'*cl; tmp2 = cr'*cr;
normR0 = sqrt(trace(tmp1*tmp2)); %!! normR0 < 1 

% define the parameters of the solver
tol=1e-6;
maxiter=100;
toltrunc=1e-14;
% iprec defines what type of preconditioning operator we are using
% iprec        vector of length 3
%              if iprec(1) equal to 1, select 1-term  preconditioner
%              if iprec(1) equal to 2, select 2-terms  preconditioner
%              else no preconditioner used
%              iprec(2:3) indexes of the PREC matrices to select
iprec=[1,1,1];      
PREC={Amean,Im};



%% subspacePCG (det)
% type_res=1, perform deterministic update of the residual (expensive in
% terms of storage allocation)
type_res=1;
% max rank of the iterates
maxrank=30;
% max rank to compute the residual matrix
maxrankR=length(OP)*maxrank;

tic
[xl,xc,xr]=sscg(OP, C, tol, maxiter, toltrunc,maxrank,maxrankR,PREC,iprec,type_res);
timeSUBCGdet = toc;

% compute the real residual norm
L=-C{1};
R=C{3};
CC=C{2};
for i=1:length(OP)
    L=[L OP{1,i}*xl];
    R=[R OP{2,i}*xr];
    CC=blkdiag(CC,xc);
end
[~,RL]=qr(L,0);
[~,RR]=qr(R,0);
clear L R
fprintf('\n\t=============================\n')
fprintf(['SubspaceCG determ \t E-time %.4f sec, Real abs. res. %e, ' ...
    'Real rel. res. %e'], timeSUBCGdet,norm(RL*CC*RR','fro'),...
    norm(RL*CC*RR','fro')/normR0)
fprintf('\n\t=============================\n')



%% subspacePCG (rand)
% type_res=2, perform randomized update of the residual (cheaper in terms
% of storage allocation)
type_res=2;
% max rank to compute the residual matrix
maxrankR=2*maxrank;
fprintf('\n')

tic
[xl,xc,xr]=sscg(OP, C, tol, maxiter, toltrunc,maxrank,maxrankR,PREC,iprec,type_res);
timeSUBCGrand = toc;

% compute the real residual norm
L=-C{1};
R=C{3};
CC=C{2};
for i=1:length(OP)
    L=[L OP{1,i}*xl];
    R=[R OP{2,i}*xr];
    CC=blkdiag(CC,xc);
end
[~,RL]=qr(L,0);
[~,RR]=qr(R,0);
clear L R
fprintf('\n\t=============================\n')
fprintf(['SubspaceCG radomized \t E-time %.4f sec, Real abs. res. %e, ' ...
    'Real rel. res. %e'], timeSUBCGrand,norm(RL*CC*RR','fro'),...
    norm(RL*CC*RR','fro')/normR0)
fprintf('\n\t=============================\n')


