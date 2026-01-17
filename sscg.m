function [Xkl, Xkc, Xkr, it] = sscg(OP, C, tol, itmax, toltrank, maxrank, maxrankR, PREC, iprec, type_res)
%
% ---------------------- Subspace CG solver -----------------------------------
%
% [Xkl,Xkc,Xkr] = sscg(OP, C, tol, itmax, toltrank, maxrank,...
% maxrankR, PREC, iprec, type_res)
%
% Solve  the multiterm matrix equation
% matrix equation formulation:
%
%         A1 X B1 + A2 X B2 + .... + Ap X Bp = rhs.
%
% INPUTS:
% OP           set of (2, p) matrices s.t. OP{1, j} = Aj and OP{2, j} = Bj
% C            set of 3 matrices such that rhs = C{1}*C{2}*C{3}'
% tol          stopping tolerance. The algorithm terminates if the relative
%              difference between iterative solutions, Xk and Xk+1, is
%              lower than tol
% itmax        max number of SsCG iterations
% maxrank      truncation value for the rank of all the matrices except the
%              residual
% maxrankR     truncation value for the rank of the residual matrix
% PREC         PREC{1, :} preconditioning matricies,
%              PREC{2, 1:2} ADI poles
% iprec        vector of length 3
%              if iprec(1) equal to 1, select 1-term  preconditioner
%              if iprec(1) equal to 2, select 2-terms  preconditioner
%              else no preconditioner used
%              iprec(2:3) indexes of the PREC matrices to select
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           FOR NO PRECONDITIONING SET PREC=[] AND iprec=3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% type_res     if equal to 1, perform deterministic update of the residual
%              if equal to 2, perform randomized update of the residual
%
% OUTPUTS:
% Xkl, Xkc, Xkr     X = Xkl*Xkc*Xkr' final approximate solution
% it                index of the last iteration performed
%
% -------------------------------------------------------------------------
% REFERENCE: 
% A subspace-conjugate gradient method for linear matrix equations
% D. Palitta, M. Iannacito, V. Simoncini
% SIAM Journal on Matrix Analysis and Applications, Vol. 46 (4), pp. 2197-2225 (2025).
% -------------------------------------------------------------------------
%
% Copyright (c): D. Palitta, M. Iannacito, V. Simoncini, January 2025.
%
% ------------------------------------------------------------------------
%  iprec(1)=1  1-term  precond   iprec(2:3) which Ai Bi coeff
%  iprec(1)=2  2-term  precond           "      "
%  iprec(1)\ne 1,2  no precond
%  type_res=1  deterministic
%  type_res=2  randomized

addpath('support_fun/');
rng(0);
% X flags
tolAB = 1e-4;                       % stopping tolerance for inner solver
maxit_cg = 1000;                    % max number of inner solver iterations
inner_direct = 4000;                % max size of the inner problem to solve directly
Xdiff_rel = 1;
p = size(OP,2);                     % number of terms in the operator


if type_res==1, disp('deterministic version'), end
if type_res==2
    disp('randomized version')
    % Omega per RSVD
    Omega_l=1/sqrt(size(C{3},1))*randn(size(C{3},1),maxrankR);
    Omega_r=1/sqrt(size(C{1},1))*randn(size(C{1},1),maxrankR);
end

% Storing arrays and initial values
it = 0;
% RHS C
Cl = C{1}; % rhs
Cc = C{2};
Cr = C{3};

% Initial guess set to 0
Xkl = zeros(size(Cl)); Xkc = zeros(size(Cc)); Xkr = zeros(size(Cr));

% Residual R0
Rkl = Cl; Rkc = -Cc; Rkr = Cr;

if  nargin>=8 && not(isempty(PREC))
    if iprec(1)==1 
        fprintf('one-term precond\n');
        ip1 = iprec(2); %ip1=input('left precond terms \n');
        ip2 = iprec(3); %ip2=input('right precond terms \n');
        PrecL = chol(PREC{1,1},'lower');
        Prec2L = chol(PREC{1,2},'lower');
        Zkl = PrecL'\(PrecL\Rkl); 
        Zkc = Rkc;
        Zkr = Prec2L'\(Prec2L\Rkr); 

    elseif iprec(1)==2
        fprintf('two-term precond\n');
        [Zkl,Zkr,Zkc] = prec_fadi_mod(PREC{1,1},PREC{1,2}, -Rkl, Rkr, Rkc,...
            PREC{2,1}, PREC{2,2},PREC{1,3},PREC{1,4});
        ip1=iprec(2);%input('l precond terms \n');
        ip2=iprec(3);%input('r precond terms \n');

    else
        % No preconditioner
        Zkl = Rkl;
        Zkc = Rkc;
        Zkr = Rkr;
        ip1=1; ip2=2;
    end

    % Directions
    [Pl,r1] = qr(Zkl,0);        % left P
    [Pr,r2] = qr(Zkr,0);
    [TP0l, LambdaP0c, TP0r] = svd(full(r1*Zkc*r2'),0);
    
else
    % No preconditioner
    % Directions
    [Pl,r1] = qr(Rkl,0);        % left P
    [Pr,r2] = qr(Rkr,0);
    [TP0l, LambdaP0c, TP0r] = svd(r1*Rkc*r2',0);
    ip1=1;ip2=1;
end

Pl = Pl*TP0l;
Pr = Pr*TP0r;
Pc = LambdaP0c;        % central P

% Start iteration
while (Xdiff_rel>tol) && (it<=itmax)

    % Compute alpha
    for i=1:p
        OP_projected{1,i} = Pl'*(OP{1,i}*Pl);
        OP_projected{1,i} = (OP_projected{1,i} + OP_projected{1,i}')/2;
        OP_projected{2,i} = Pr'*(OP{2,i}*Pr);
        OP_projected{2,i} = (OP_projected{2,i} + OP_projected{2,i}')/2;
    end

    % rhs for the smaller problem
    RHSl = Pl'*Rkl;   RHSc = Rkc;    RHSr = Pr'*Rkr;
    RHS = RHSl*RHSc*(RHSr');
    rhs = -RHS(:);

    % call inner solver
    if numel(RHS)<inner_direct
        % direct solver through Kronecker transformations
        optot=0;
        for i=1:p
            optot=optot+kron(OP_projected{2,i},OP_projected{1,i});
        end
        optot=(optot+optot')/2;
        Coptot=chol(optot,'lower');
        uk=Coptot'\(Coptot\rhs); 
    else
        % iterative solution
        if iprec(1)==2
            nh=size(OP_projected{1,ip1},1);
            [Prec1.VP,Prec1.EP]=eig(OP_projected{1,ip1});
            [Prec2.VP,Prec2.EP]=eig(OP_projected{2,ip2});
            Prec1.L=diag(Prec1.EP)*ones(1,nh)+ones(nh,1)*diag(Prec2.EP).';
            [uk,~] = pcg_inner(OP_projected,rhs,0*rhs,maxit_cg,...
                tolAB,Prec1,Prec2,iprec(1));
        else
            [uk,~] = pcg_inner(OP_projected,rhs,0*rhs,maxit_cg,tolAB,OP_projected{1,ip1},OP_projected{2,ip2},iprec(1));
        end
    end % end inner solver


    Uk = reshape(uk, [size(Pl,2), size(Pr,2)]);     % alpha_k

    % update the iterative solution
    if it==0
        Xkl = Pl;     Xkc=Uk;      Xkr = Pr;
    else
        Xkc=blkdiag(Uk,Xkc);
    end

    % Truncate the updated solution
    if it>0
        [QXl, RXl] = qr([Pl,Xkl],0);
        [QXr, RXr] = qr([Pr,Xkr],0);

        [TXkl, LambdaXkc, TXkr] = svd(RXl*Xkc*RXr',0);
        ll=diag(LambdaXkc);
        index_svd=min(find(ll/ll(1)>=toltrank,1,'last'),maxrank);

        Xkl = QXl*TXkl(:,1:index_svd);
        Xkc = LambdaXkc(1:index_svd,1:index_svd);
        Xkr = QXr*TXkr(:,1:index_svd);
        clear QXl RXl QXr RXr

        Xdiff_abs=norm(Uk,'fro');
        Xdiff_rel=Xdiff_abs/norm(Xkc,'fro');
        % check stopping criterion
        if Xdiff_rel<tol
            fprintf('abs and rel X diff %d %d\n',Xdiff_abs,Xdiff_rel),
            fprintf('%d\t %.4e\t  %d\t %d \t %d\n',[it, Xdiff_rel, min(size(Xkc)), ...
                min(size(Pc)), min(size(Rkc))])
            break
        end
    end % end trunc X

    % Update 'quasi'-residual matrix
    if type_res==1              % deterministic 'quasi'-residual update
        
        [Rkl, Rkc, Rkr] = update_determin(Cl, Cc, Cr, OP, Xkl, Xkc, Xkr,...
            toltrank, maxrank);

    elseif type_res==2          % randomized 'quasi'-residual update
        
        [Rkl,Rkc,Rkr]=update_rsvd(Cl, Cc, Cr, OP, Xkl, Xkc, Xkr, ...
            toltrank, maxrank, Omega_l, Omega_r);
   
    end % end 'quasi'-residual update

    % Apply Preconditioner
    if iprec(1)==1  % 1-term precond
        Zkl=PrecL'\(PrecL\Rkl);
        Zkc=Rkc;
        Zkr = Prec2L'\(Prec2L\Rkr);
    
    elseif iprec(1)==2  % 2-term precond
        [Zkl,Zkr,Zkc] = prec_fadi_mod(PREC{1,1},PREC{1,2}, Rkl, Rkr, -Rkc,...
            PREC{2,1}, PREC{2,2},PREC{1,3},PREC{1,4});

        [nz1,nz2]=size(Zkl);
        if nz2>nz1     % fadi does not check dimensions
            [uz1,sz1,vz1]=svd(Zkl,0);
            [uz2,sz2,vz2]=svd(Zkr,0);
            Zkl=uz1*sz1(:,1:nz1);
            Zkr=uz2*sz2(:,1:nz1);
            Zkc=vz1(:,1:nz1)'*Zkc*vz2(:,1:nz1);
        end
    else
        Zkl=Rkl;     Zkc=Rkc;      Zkr=Rkr;
    end
    
    % Compute beta_k
    % compute rhs for smaller inner problem
    RHSl=cell(1,p);
    RHSr=cell(1,p);

    for i=1:p
        RHSl{i}=Pl'*(OP{1,i}*Zkl)*Zkc;
        RHSr{i}=Pr'*(OP{2,i}*Zkr);
    end
    RHS = cell2mat(RHSl)*cell2mat(RHSr)';

    rhs = -RHS(:);

    % call inner solver
    if numel(RHS)<inner_direct
        vk=Coptot'\(Coptot\rhs); 
    else
        if iprec(1)==2
            [vk,~] = pcg_inner(OP_projected,rhs,rhs*0,maxit_cg,...
                tolAB,Prec1,Prec2,iprec(1));
        else
            [vk,~] = pcg_inner(OP_projected,rhs,0*rhs,maxit_cg,...
                tolAB,OP_projected{1,ip1},OP_projected{2,ip2},iprec(1));
        end

    end % end inner solver

    Vk = reshape(vk, [size(Pl,2), size(Pr,2)]);     % beta_k

    clear vk

    % Update and truncate directions
    if  nargin>=8 && not(isempty(PREC))
        [QPl,RPl]=qr([Pl,Zkl],0);
        [QPr,RPr]=qr([Pr,Zkr],0);
        [TPkl, LambdaPkc, TPkr] = svd(RPl*blkdiag(Vk,Zkc)*(RPr'),0);

    else
        [QPl,RPl]=qr([Rkl,Pl],0);
        [QPr,RPr]=qr([Rkr,Pr],0);
        [TPkl, LambdaPkc, TPkr] = svd(RPl*blkdiag(Rkc, Vk)*(RPr'),0);

    end

    ll=diag(LambdaPkc);
    index_svd=min(find(ll/ll(1)>=toltrank,1,'last'),maxrank);

    Pl = QPl*TPkl(:,1:index_svd);
    Pc = LambdaPkc(1:index_svd,1:index_svd);
    Pr = QPr*TPkr(:,1:index_svd);
    clear QPl QPr RPl RPr
    % end directions update and truncation
    
    % display current iteration info
    fprintf('%d\t %.4e\t  %d\t %d \t %d\n',[it, Xdiff_rel, min(size(Xkc)), ...
        min(size(Pc)), min(size(Rkc))])

    it=it+1;
end % end of while, outer sscg iterations 
end

% Residual update functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [Rkl,Rkc,Rkr] = update_rsvd(Cl, Cc, Cr, OP, Xkl, Xkc, Xkr, ...
            toltrank, maxrank, Omega_l, Omega_r)

        term1 = -(Cl*(Cc*(Cr'*Omega_l)));
        term1T = -(Cr*(Cc'*(Cl'*Omega_r)));
        for i=1:length(OP)
            wrk1=OP{1,i}*Xkl;
            wrk2=OP{2,i}*Xkr;
            term1=term1+wrk1*(Xkc*(wrk2'*Omega_l));
            term1T=term1T+wrk2*(Xkc'*(wrk1'*Omega_r));
        end
        clear wrk1 wrk2
        [Q,~]=qr(term1,0);
        [QT,~]=qr(term1T,0);
        clear term1 term1T

        term2=-((Q'*Cl)*Cc)*(Cr'*QT);
        for i=1:length(OP)
            term2=term2+((Q'*(OP{1,i}*Xkl))*Xkc)*((OP{2,i}*Xkr)'*QT);
        end
        [UU,SS,VV] = svd(term2,0);
        clear term2

        ll=diag(SS);
        index_svd=min(find(ll/ll(1)>=toltrank,1,'last'),maxrank);
        Rkl = Q*(UU(:,1:index_svd));
        Rkr = QT*VV(:,1:index_svd);

        Rkc = SS(1:index_svd,1:index_svd);
    end % end of update_rsvd


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [Rkl,Rkc,Rkr] = update_determin(Cl, Cc, Cr, OP, Xkl, Xkc, Xkr, ...
        toltrank, maxrank)

    % Compute L(X)
    AX1{1}=OP{1,1}*(Xkl*Xkc);
    AX2{1}=OP{2,1}*Xkr;
    for k=2:length(OP)
        AX1{k}=OP{1,k}*(Xkl*Xkc);
        AX2{k}=OP{2,k}*Xkr;
    end

    Rkl = [Cl cell2mat(AX1)];
    Rkc = blkdiag(-Cc,speye(length(OP)*size(Xkr,2)));
    Rkr = [Cr cell2mat(AX2)];

    % Prepare for truncation
    [QRl, RRl] = qr(Rkl, 0);
    [QRr, RRr] = qr(Rkr, 0);

    res = RRl*Rkc*RRr';

    % Truncate the updated residual
    [TRkl, LambdaRkc, TRkr] = svd(res,0);
    ll=diag(LambdaRkc);
    index_svd=min(find(ll/ll(1)>=toltrank,1,'last'),maxrank);

    Rkl = QRl*TRkl(:,1:index_svd);
    Rkc = LambdaRkc(1:index_svd,1:index_svd);
    Rkr = QRr*TRkr(:,1:index_svd);
    end % end of update_determin

