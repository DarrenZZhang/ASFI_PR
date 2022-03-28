clear all; clc; close all; clear memory;

load FERET_40x40
addpath 'utility'

fea = double(fea);
%% Image processing term for all the data
fea_hat=zeros(row*col,length(gnd));
for i=1:length(gnd)
    % Generate naive Left and Right samples  
    tmpVec=fea(:,i);
    tmpImg=reshape(tmpVec,row,col);               
    trLface(1:row,1:col/2)= tmpImg(1:row,1:col/2);
    trRface(1:row,col/2+1:col)= tmpImg(1:row,col/2+1:col);   
    for km=1:col/2
        virRface(:,km)=trRface(:,col-(km-1));            
    end
    tr_z1=reshape(trLface(:,1:col/2),row*col/2,1);
    tr_z2=reshape(virRface(:,1:col/2),row*col/2,1);
    
    % Generate approximate symmetrical face images from naive virtual training samples     
    max_iter = 30;
    for kmn = 1:max_iter
        grad1 = tr_z1-tr_z2;
        tr_z1 = tr_z1-0.3/kmn*grad1;
        grad2 = tr_z2-tr_z1;
        tr_z2 = tr_z2-0.3/kmn*grad2;
        if norm(tr_z1-tr_z2) < 1.0e-2
            break;
        end 
    end

    virRface = reshape(tr_z2,row,col/2);
    optiR = zeros(size(virRface));
    for km=1:col/2            
        optiR(:,km) = virRface(:,col/2-km+1);
    end 
    optiRvec = reshape(optiR,row*col/2,1);
    virTr=[tr_z1;optiRvec];
    fea_hat(:,i)=virTr(:)/norm(virTr);
end

clearvars -except fea_hat gnd row col

%% image classification term using NN classifier
for train_num = 1:5
    class_num = length(unique(gnd)); % Number of classes
    numClass = zeros(class_num,1);
    for i=1:class_num
        numClass(i,1) = length(find(gnd==i));
    end
    
    trfea = []; ttfea = []; 
    trgnd = []; ttgnd = [];
    for j = 1:class_num
        index = find(gnd == j); 
        randIndex = 1:numClass(j);
        trfea = [trfea fea_hat(:,index(randIndex(1:train_num)))];
        trgnd = [trgnd ; gnd(index(randIndex(1:train_num)))];
        ttfea = [ttfea fea_hat(:,index(randIndex(train_num+1:end)))];
        ttgnd = [ttgnd ; gnd(index(randIndex(train_num+1:end)))];
    end
    
    % obtain the classification acc
    [acc,label] = NNClassifier_L1(trfea,ttfea,trgnd',ttgnd');
    fprintf('Num_train is %d, Acc is %.2f%% \n',train_num,acc);
    clearvars -except train_num fea_hat gnd row col
end
 