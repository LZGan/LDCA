clc;
clear all;
close all;

load c.mat;
load Label.mat
tic
%mri_ad=c(1:186,1:93);
mri_mci=c(187:579,1:93);
mri_nc=c(580:805,1:93);
%mri=[mri_ad;mri_nc];
%mriam=[mri_ad;mri_mci];
mrimn=[mri_mci;mri_nc];

%label_ad=Label(1:186);
%label_nc=Label(580:805);
%label=[label_ad;label_nc];
%label1=Label(1:579)
label2=Label(187:805);

%pet_ad=c(1:186,94:186)
pet_mci=c(187:579,94:186);
pet_nc=c(580:805,94:186);
%pet=[pet_ad;pet_nc];
%petam=[pet_ad;pet_mci];
petmn=[pet_mci;pet_nc];

%mri=solve_lrr(mri',mri',0);
%pet=solve_lrr(pet',pet',0);

%mriam=solve_lrr(mriam',mriam',0);
%petam=solve_lrr(petam',petam',0);

mrimn=solve_lrr(mrimn',mrimn',0);
petmn=solve_lrr(petmn',petmn',0);

%pet=pet'
%mri=mri'

%petam=petam'
%mriam=mriam'

petmn=petmn'
mrimn=mrimn'

label=label2;
[Ax,Ay,Xs,Ys] = dcaFuse(mrimn,petmn,label);
fusionmp=[Xs' Ys'];
%label=label2;
save fusionmp
data=fusionmp;
[m,n]=size(data);
indices =crossvalind('Kfold', m, 10);
for i=1:10 
  test=(indices==i); 
  train=~test; 
  data_train=data(train,:); 
  label_train=label(train,:); 
  data_test=data(test,:); 
  label_test=label(test,:); 
  test=(indices==i);  
end

model =svmtrain(label_train,data_train,'-c 1 -t 2');
[predict_label, accuracy, dec_values] = svmpredict(label_test,data_test,model);
%[ACC,Sen,Spe,BAC,PPV,NPV,Fmeasure,MCC,Gmeasure]=lmx_SenSpeAcc(predict_label,label_test)
plotSVMroc(label_test,predict_label,2);
[X,Y,T,AUC] = perfcurve(label_test,scores,posclass)
toc