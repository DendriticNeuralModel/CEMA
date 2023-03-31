function O = evaluate_accuracy( x, pop, M )

[II,J] = size(x);
[G,~] = size(pop);
k = 5;
O = zeros(J,G);

for g=1:G
    
    w=zeros(II,M);                  % the weight value of dendrites
    q=zeros(II,M);                  % the threshold value of dendrites
    
    for m=1:M
        w(:,m)= pop(g,(1+2*II*(m-1)):(II+2*II*(m-1)))';
        q(:,m)= pop(g,(1+II+2*II*(m-1)):(2*m*II))';
    end
    Y=zeros(II,M,J);
    Z=ones(M,J);
    V=zeros(1,J);
    
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
    end

end

