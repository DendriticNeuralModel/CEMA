function Metrics=CEMA(input_train,target_train,K,input_test,target_test,M)
                                             %% training cost-sensitive DNMs %%
%Parameters                                           
subpopsize=50;
Max_Gen=1500;
% Set the scaling factor and crossover control parameter of DE
Fmin=0.4;
Fmax=1.2;
CR = 0.85;
c=0.8;   %the coefficient of penalty term
l=0.8;   %the coefficient of regularity term
p=0.5;   %the probability of sampling archive


D=2*M*size(input_train,1);
lu=[-2*ones(1,D); 2*ones(1,D)];
% Initialize the population
Pop = repmat(lu(1, :), subpopsize*K, 1) + rand(subpopsize*K, D) .* (repmat(lu(2, :) - lu(1, :), subpopsize*K, 1));
fit=zeros(1,subpopsize*K);
BestIndex=zeros(1,K);
for k=1:K
    targettrain_class=target_train(k,:);
    %Select randomly an individual from each class
    Part_index=randi(subpopsize,1,K);
    Part_index=[0:K-1]*subpopsize+Part_index;
    Part_index(k)=[];
    Pop2=Pop(Part_index,:);
    %Evaluate the full solutions using the fitness function
    Pop1=Pop((k-1)*subpopsize+1 : k*subpopsize,:);
    fit((k-1)*subpopsize+1 : k*subpopsize)=PenalFitness(Pop1, Pop2, input_train, targettrain_class, M, c, l);
    [~,BestIndex(1,k)]=min(fit((k-1)*subpopsize+1 : k*subpopsize));
    BestIndex(1,k)=BestIndex(1,k)+(k-1)*subpopsize;
end
%Set the initial difference vector archive
Archive=cell(1,K);
for ita=1:Max_Gen
    F = (Fmax-Fmin)*((Max_Gen-ita)/Max_Gen)+Fmin;
    for k=1:K
        targettrain_class=target_train(k,:);
        %Select the best individual from each population
        Part_index=BestIndex;
        Part_index(k)=[];
        Pop2=Pop(Part_index,:);
       %% Apply evolutionary operators to create an offspring
        Pop1=Pop((k-1)*subpopsize+1 : k*subpopsize,:);
        [r1, r2, r3] = getindex3(subpopsize);
        DE=Pop1(r2, :) - Pop1(r3, :);
        if ita>=2 && ~isempty(Archive{1,k})
            l_rand=rand(1,subpopsize)<=p;
            SampleSize=sum(l_rand);
            S_index=randi(size(Archive{1,k},1),1,SampleSize);
            DE(l_rand,:)=Archive{1,k}(S_index,:);
        end
        % Implement DE/rand/1 mutation
        V = Pop1(r1, :) + F * DE;
        V = repair(V, lu);
        
        % Implement binomial crossover
        Offspring=zeros(subpopsize,D);
        for i = 1:subpopsize
            j_rand = floor(rand * D) + 1;
            t = rand(1, D) < CR;
            t(1, j_rand) = 1;
            t_ = 1 - t;
            Offspring(i, :) = t .* V(i, :) + t_ .* Pop1(i, :);
        end
        OffFit=PenalFitness(Offspring, Pop2, input_train, targettrain_class, M, c, l);
        ParentFit=PenalFitness(Pop1, Pop2, input_train, targettrain_class, M, c, l);
        %selection
        for i = 1:subpopsize
            if OffFit(i) < ParentFit(i)
                Pop((k-1)*subpopsize+i, :) = Offspring(i, :);
                ParentFit(i) = OffFit(i);
                %Store the successful difference vector into archive.
                Archive{1,k}=[Archive{1,k};DE(i,:)];
            end
        end
        %address archive
        ArcSize=size(Archive{1,k},1);
        if ArcSize>subpopsize
            Archive{1,k}( randperm(ArcSize,ArcSize-subpopsize) ,:)=[];
        end
        [~,BestIndex(1,k)]=min(ParentFit);
        BestIndex(1,k)=BestIndex(1,k)+(k-1)*subpopsize;       
    end
end
Otrain = evaluate_accuracy( input_train, Pop(BestIndex,:), M );
Otest = evaluate_accuracy( input_test, Pop(BestIndex,:), M );
Metrics=zeros(2,2);
%DNM-train
CM=zeros(K,K);
TrainSize=size(target_train,2);
r=Otrain';
[~,I] = max(r);
for x=1:TrainSize
    if target_train(I(x),x)==1
        CM(I(x),I(x))=CM(I(x),I(x))+1;
    else
        [~,maxindex]=max(target_train(:,x));
       CM(maxindex,I(x))=CM(maxindex,I(x))+1; 
    end
end
[Metrics(1,1),Metrics(1,2)]=Acc_kappa(CM,K,TrainSize);

%DNM-test
CM=zeros(K,K);
TestSize=size(target_test,2);
r=Otest';
[~,I] = max(r);
for x=1:TestSize
    if target_test(I(x),x)==1
        CM(I(x),I(x))=CM(I(x),I(x))+1;
    else
        [~,maxindex]=max(target_test(:,x));
       CM(maxindex,I(x))=CM(maxindex,I(x))+1; 
    end
end
[Metrics(2,1),Metrics(2,2)]=Acc_kappa(CM,K,TestSize);
end

function fit=PenalFitness(Pop1, Pop2, input, target, M, c, l)
    %Pop1 --- classifers from r-th class
    %Pop2 --- classifers with the best fitness value from other classes 
    [fitness,O1] = evaluate_objective( input, target, Pop1, M);%1xpopsize
    Temp=target==1;
    [~,O2] = evaluate_objective( input(:,Temp), target(:,Temp), Pop2, M);
    O2=max(O2,[],2);
    O1=O1(Temp,:);
    Po=O2-O1;
    Po(Po<0)=0;
    Po=mean(Po.^2);
    L2=l*(1./sqrt(sum(Pop1.^2,2)));
    fit=fitness+c*Po+L2';
end
  