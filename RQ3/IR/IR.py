import sys
import os
from tqdm import tqdm
def getJaccard(list1, list2):
    list1 = list(set(list1))
    list2 = list(set(list2))
    intersectionList = []
    for l in list1:
        if l in list2:
            intersectionList.append(l)
    tempList = []
    tempList.extend(list1)
    tempList.extend(list2)
    unionList = list(set(tempList))
    return len(intersectionList) * 1.0 / len(unionList)


def getDice(list1, list2):
    list1 = list(set(list1))
    list2 = list(set(list2))
    denominator = len(list1) + len(list2)
    unionList = []
    for l in list1:
        if l in list2:
            unionList.append(l)
    return len(unionList) * 2.0 / denominator


def getOverlap(list1, list2):
    list1 = list(set(list1))
    list2 = list(set(list2))
    denominator = min(len(list1), len(list2))
    unionList = []
    for l in list1:
        if l in list2:
            unionList.append(l)
    return len(unionList) * 1.0 / denominator


def IR(trainFocalMethodFile, trainAssertionFile, testFocalMethodFile, testsAssertionFile):
    fTrainFocalMethod = open(trainFocalMethodFile, 'r', encoding="utf-8")
    fTestFocalMethod = open(testFocalMethodFile, 'r', encoding="utf-8")
    trainFocalMethodList = fTrainFocalMethod.read().rstrip("\n").split("\n")
    testFocalMethodList = fTestFocalMethod.read().rstrip("\n").split("\n")
    fTrainFocalMethod.close()
    fTestFocalMethod.close()

    fTrainAssertion = open(trainAssertionFile, 'r', encoding="utf-8")
    fTestAssertion = open(testsAssertionFile, 'r', encoding="utf-8")
    trainAssertionList = fTrainAssertion.read().rstrip("\n").split("\n")
    testAssertionList = fTestAssertion.read().rstrip("\n").split("\n")
    fTrainAssertion.close()
    fTestAssertion.close()

    trainFocalTestMethodListSplit = []
    for i in range(0, len(trainFocalMethodList)):
        tempList = trainFocalMethodList[i].split(" ")
        trainFocalTestMethodListSplit.append(tempList)
    fRecordIRResult = open(os.path.join(output_path, "new_evosuite_IRResultTest.txt"), 'w', encoding="utf-8")
    #IRResult = open(os.path.join(output_path, "IR.txt"), 'w', encoding="utf-8")
    retrieval_save = open(os.path.join(output_path, "new_evosuite_retrieval_test_saved"),"w+")
    correctNum = 0
    totNum = len(testFocalMethodList)
    # len(testFocalMethodList)
    for i in tqdm(range(len(testFocalMethodList))):
        tempTestFocalList = testFocalMethodList[i].split(" ")
        maxStep = 0
        maxValue = -1
        for j in range(0, len(trainFocalTestMethodListSplit)):
            jaccard = getJaccard(tempTestFocalList, trainFocalTestMethodListSplit[j])
            if jaccard > maxValue:
                maxValue = jaccard
                maxStep = j
        fRecordIRResult.write(
            trainFocalMethodList[maxStep] + "\n" + testFocalMethodList[i] + "\n" + trainAssertionList[maxStep] + "\n" +
            testAssertionList[i] + "\n\n")
        retrieval_save.write(str(maxValue)+ ' ' + str(maxStep) + '\n')
        #IRResult.write(trainAssertionList[maxStep]+'\n')
        if trainAssertionList[maxStep] == testAssertionList[i]:
            correctNum += 1
    fRecordIRResult.close()
    return correctNum * 1.0 / totNum
def IR_Train(trainFocalMethodFile, trainAssertionFile):
    fTrainFocalMethod = open(trainFocalMethodFile, 'r', encoding="utf-8")
    trainFocalMethodList = fTrainFocalMethod.read().rstrip("\n").split("\n")
    fTrainFocalMethod.close()
    fTrainAssertion = open(trainAssertionFile, 'r', encoding="utf-8")
    trainAssertionList = fTrainAssertion.read().rstrip("\n").split("\n")
    fTrainAssertion.close()

    trainFocalTestMethodListSplit = []
    for i in range(0, len(trainFocalMethodList)):
        tempList = trainFocalMethodList[i].split(" ")
        trainFocalTestMethodListSplit.append(tempList)
    fRecordIRResult = open(os.path.join(output_path, "IRResultTrain.txt"), 'w', encoding="utf-8")
    retrieval_save = open(os.path.join(output_path, "retrieval_train_saved"),"w+")
    correctNum = 0
    totNum = len(trainFocalTestMethodListSplit)
    #
    for i in tqdm(range(len(trainFocalTestMethodListSplit))):
        tempTestFocalList = trainFocalTestMethodListSplit[i]
        maxStep = 0
        maxValue = -1
        for j in range(0, len(trainFocalTestMethodListSplit)):
            if i == j:
                continue
            jaccard = getJaccard(tempTestFocalList, trainFocalTestMethodListSplit[j])
            if jaccard > maxValue:
                maxValue = jaccard
                maxStep = j
        fRecordIRResult.write(
            trainFocalMethodList[maxStep] + "\n" + trainFocalMethodList[i] + "\n" + trainAssertionList[maxStep] + "\n" +
            trainAssertionList[i] + "\n\n")
        retrieval_save.write(str(maxValue)+ ' ' + str(maxStep) + '\n')
        if trainAssertionList[maxStep] == trainAssertionList[i]:
            correctNum += 1
    fRecordIRResult.close()
    return correctNum * 1.0 / totNum
#def read_data(data_path):
#    with open(data_path) as f:
#        paths = f.read().split('\n')
#    train_method = paths[0]
#    test_method = paths[1]
#    train_assert = paths[2]
#    test_assert = paths[3]
#    valid_method = paths[4]
#    valid_assert = paths[5]
#    old_test_method = paths[6]
#    old_test_assert = paths[7]
#    return train_method, test_method, train_assert, test_assert,valid_method,valid_assert,old_test_method,old_test_assert
def read_data(data_path):
    with open(data_path) as f:
        paths = f.read().split('\n')
    train_method = paths[0]
    train_assert = paths[1]
    test_method = paths[2]
    test_assert = paths[3]
    return train_method,train_assert,test_method,test_assert
if __name__ == '__main__':
    config = sys.argv[1]
    output_path = sys.argv[2]
#    trainFocalMethodFile,  testFocalMethodFile, trainAssertionFile, testsAssertionFile,validFocalMethodFile,validAssertionFile,oldtestFocalMethodFile,oldtestsAssertionFile= read_data(config)
    trainFocalMethodFile,trainAssertionFile,testFocalMethodFile, testsAssertionFile= read_data(config)
    # IR(trainFocalMethodFile, trainAssertionFile, testFocalMethodFile, testsAssertionFile)
    # IR_Train(trainFocalMethodFile, trainAssertionFile)
    IR(trainFocalMethodFile, trainAssertionFile, testFocalMethodFile, testsAssertionFile)