clc
clear all
tic

F_index=1;                     % Problem number
divide_rate=0.7;               % The rate of dividing the dataset

%read and normialize data
[input_train, target_train, input_test, target_test, denNumber] = divideDataset(F_index, divide_rate);
[input_train, PS]=mapminmax(input_train,0,1);
input_test = mapminmax('apply',input_test,PS);
K=size(target_train,1);

Metrics=CEMA(input_train,target_train,K,input_test,target_test,denNumber);        
Acc_train=Metrics(1,1);
Kappa_train=Metrics(1,2);
Acc_test=Metrics(2,1);
Kappa_test=Metrics(2,2);
        
toc;



