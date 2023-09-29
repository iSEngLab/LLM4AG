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
    df.to_csv("assert_{}_old.csv".format(name),index=False,escapechar="\\")
if __name__ == '__main__':
    # transTxt("train","./OldDataSet/Training/testMethods.txt","./OldDataSet/Training/assertLines.txt")
    # transTxt("val", "./OldDataSet/Validation/testMethods.txt", "./OldDataSet/Validation/assertLines.txt")
    # transTxt("test", "./OldDataSet/Testing/testMethods.txt", "./OldDataSet/Testing/assertLines.txt")
    df1 = pd.read_csv('kotlin_data_10000_train.csv')
    df2 = pd.read_csv('kotlin_data_5000_train.csv')
    print(len(df1))
    print(len(df2))
