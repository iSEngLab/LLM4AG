import csv

import pandas as pd
import torch


def transTxt(name,path1,path2):
    f1 = open(path1,'r',encoding='utf-8')
    f2 =  open(path2,'r',encoding='utf-8')
    source = f1.readlines()
    target = f2.readlines()
    source = [s.strip() for s in source]
    target = [t.strip() for t in target]
    df = pd.DataFrame()
    df['source'] = source
    df['target'] = target
    df.to_csv("assert_{}.csv".format(name),index=False,escapechar="\\")
if __name__ == '__main__':
    transTxt("train","./NewDataSet/Training/testMethods.txt","./NewDataSet/Training/assertLines.txt")
    transTxt("val", "./NewDataSet/Validation/testMethods.txt", "./NewDataSet/Validation/assertLines.txt")
    transTxt("test", "./NewDataSet/Testing/testMethods.txt", "./NewDataSet/Testing/assertLines.txt")