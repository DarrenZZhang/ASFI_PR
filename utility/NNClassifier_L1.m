function [acc,PreLabel] = NNClassifier_L1(Samples_Train,Samples_Test,Labels_Train,Labels_Test)
Train_Model = Samples_Train;
Test_Model = Samples_Test;
numTest = size(Test_Model,2);
numTrain = size(Train_Model,2);

PreLabel = [];
for test_sample_no = 1:numTest
    testMat = repmat(Test_Model(:,test_sample_no), 1, numTrain);
    scores_vec = sum(abs(testMat - Train_Model), 1);
    [min_val min_idx] = min(scores_vec);
    best_label = Labels_Train(1,min_idx);
    PreLabel = [PreLabel, best_label];
end

Comp_Label = PreLabel - Labels_Test;
acc = (sum((Comp_Label==0))/numel(Comp_Label))*100;

end
            
            