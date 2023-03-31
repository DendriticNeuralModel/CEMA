function [ACC,Kappa]=Acc_kappa(CM,K,test_size)
TR=zeros(1,K);
TC=zeros(1,K);
TRC=0;
H=0;
n=test_size;
for i=1:K
    for j=1:K
        TR(1,i)=TR(1,i)+CM(i,j);
        TC(1,j)=TC(1,j)+CM(i,j);
        if i==j
            H=H+CM(i,i);
        end
    end
end
for i=1:K
    TRC=TRC+TR(1,i)*TC(1,i);
end
ACC=H/n;
Kappa=(n*H-TRC)/(n*n-TRC);