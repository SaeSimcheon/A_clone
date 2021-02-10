from Simple_logistic_regression import Utils

print(Utils.add(1,2))

trainX,testX,trainY,testY=Utils.sample_data_call("./data_set/heart/heart.csv")

Utils.normalize(trainX,testX)