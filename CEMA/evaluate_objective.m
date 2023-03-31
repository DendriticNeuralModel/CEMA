function [fitness,O] = evaluate_objective( x, T, pop, M)
%consider cost-sensitive operator for solving the imblanced problem
[II,J] = size(x);
[popsize,~] = size(pop);
k = 5;
O = zeros(J,popsize);      
fitness = zeros(1,popsize);
N_1=sum(T);%the number of positive samples
N_0=J-N_1;%the number of negative samples
for g=1:popsize
    
    w=zeros(II,M);                  % the weight value of dendrites
    q=zeros(II,M);                  % the threshold value of dendrites
    
    for m=1:M
        w(:,m)= pop(g,(1+2*II*(m-1)):(II+2*II*(m-1)))';
        q(:,m)= pop(g,(1+II+2*II*(m-1)):(2*m*II))';
    end
    Y=zeros(II,M,J);
    Z=ones(M,J);
    V=zeros(1,J);
    E=zeros(1,J);
    
    for j=1:J
        % build synaptic layers
        for m=1:M
            for i=1:II
                Y(i,m,j)=1/(1+exp(-k*(w(i,m)*x(i,j)-q(i,m))));
            end
        end
        
        % build dendrite layers
        for m=1:M
            Q=1;
            for i=1:II
                Q=Q*Y(i,m,j);
            end
            Z(m,j)=Q;
        end
        
        % build  membrane layers
        constant=0;
        for m=1:M
            constant=constant+Z(m,j);
        end
        V(j)=constant;
        
        % build a soma layer
        O(j,g)=1/(1+exp(-k*(V(j)-0.5)));
        if T(j)==1
            cost=N_0/J;
        else
            cost=N_1/J;
        end
        E(j)=cost*((O(j,g)-T(j))^2);
    end
    
    % calculate the fitness
    fitness(1,g)=sum(E);
end