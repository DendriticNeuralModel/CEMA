% Start
function [ input_train, target_train, input_test, target_test, denNumber ] = divideDataset( F_index, divide_rate)
switch  F_index
    case 1
        load iris_dataset
        input=irisInputs;
        target=irisTargets;
        denNumber=8;
    case 2
        load wine_dataset
        input=wineInputs;
        target=wineTargets;
        denNumber=10;   
end
K=size(target,1);
input_train=[];
target_train=[];
input_test=[];
target_test=[];
for k=1:K
    inputk=input(:,target(k,:)==1);
    targetk=target(:,target(k,:)==1);
    r=size(targetk,2);
    if r<=10
        A=1:r;
        C=1:r;
    else
        [A,~,C] = dividerand(r,divide_rate,0,1-divide_rate);
    end
    input_train=[input_train inputk(:,A)];         input_test=[input_test inputk(:,C)];
    target_train=[target_train targetk(:,A)];       target_test=[target_test targetk(:,C)];
end
end
% Over

