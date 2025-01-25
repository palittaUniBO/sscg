function [A,N,B,C] = Heat_Transfer_Model(k,left,upper,right,lower,d)
% x_t= delta(x)
% (a) Dirichlet BC: x = 0
% (b) Dirichlet BC: x = u
% (c) Robin BC: n * nabla(x) = d * u * (x-1) 
   
    
h=1/(k+1);
T=sparse(k);
T(1,1)=-2;
T(1,2)=1;
for i=2:k-1
    T(i,i-1)=1;
    T(i,i)=-2;
    T(i,i+1)=1;
end
T(k,k-1)=1;
T(k,k)=-2;

I=speye(k);

E1=I(:,1)*I(1,:);
Ek=I(:,k)*I(k,:);
A=1/h^2*(kron(I,T) + kron(T,I));

switch left
    case 'a'
        B=[];
        N=[];
    case 'b'
        B=1/h*kron(I(:,1),ones(k,1));
        N=[];
    case 'c'
        A=A+d*1/h^2*kron(E1,I);
        B=d/h*kron(I(:,1),ones(k,1));
        N=reshape(-d/h*kron(E1,I),k^4,1);
    otherwise
        B=[];
        N=[];
end

switch upper
    case 'a'
        
    case 'b'
        B=[B 1/h*kron(ones(k,1),I(:,1))];
    case 'c'
        A=A+d/h^2*kron(I,E1);
        B=[B d/h*kron(ones(k,1),I(:,1))];
        N=[N reshape(-d/h*kron(I,E1),k^4,1)];
    otherwise
        
end

switch right
    case 'a'
        
    case 'b'
        B=[B 1/h*kron(I(:,k),ones(k,1))];
    case 'c'
        A=A+d/h^2*kron(Ek,I);
        B=[B d/h*kron(I(:,k),ones(k,1))];
        N=[N reshape(-d/h*kron(Ek,I),k^4,1)];
    otherwise
        
end

switch lower
    case 'a'
        
    case 'b'
        B=[B 1/h*kron(ones(k,1),I(:,k))];
    case 'c'
        A=A+d/h^2*kron(I,Ek);
        B=[B d/h*kron(ones(k,1),I(:,k))];
        N=[N reshape(-d/h*kron(I,Ek),k^4,1)];
    otherwise
end
C=1/k^2*ones(1,k^2);
N=reshape(N,size(A,1),[]);
B=full(B);                  % added by you