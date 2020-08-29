from DataInterface import getExtraCreditDataset
from DecisionTree import makeTree, setEntropy,infoGain
from Testing import getAverageClassificaionRate, printDemarcation



def testExtraCreditDataset(setFunc = setEntropy, infoFunc = infoGain):
    examples,attrValues,labelName,labelValues = getExtraCreditDataset()
    print 'Testing Breast Cancer dataset. Number of examples %d.'%len(examples)
    tree = makeTree(examples, attrValues, labelName, setFunc, infoFunc)
    f = open('breast-cancer.out','w')
    f.write(str(tree))
    f.close()
    print 'Tree size: %d.\n'%tree.count()
    print 'Entire tree written out to breast-cancer.out in local directory\n'
    dataset = getExtraCreditDataset()
    evaluation = getAverageClassificaionRate((examples,attrValues,labelName,labelValues))
    print 'Results for training set:\n%s\n'%str(evaluation)
    printDemarcation()
    return (tree,evaluation)

testExtraCreditDataset()
