import os

import pandas as pd


def getStatCsv(type):
    path = './{}_rq2'.format(type)
    bugs = []
    model_bugs = {}
    for dir in os.listdir(path):
        model = dir.split('_')[0]
        if not model in model_bugs.keys():
            model_bugs[model]=[]
        for file in os.listdir(path+'/'+dir):
            f = open(path+'/'+dir+'/'+file,'r')
            bs = f.read().splitlines()
            bugs+=bs
            model_bugs[model]+=bs
        model_bugs[model] = set(model_bugs[model])
    bugs = sorted(set(bugs))
    model_bugs_detect = {}
    for model in model_bugs.keys():
        if not model in model_bugs_detect.keys():
            model_bugs_detect[model]=[]
        model_bug = model_bugs[model]
        for b in bugs:
            if b in model_bug:
                model_bugs_detect[model].append(1)
            else:
                model_bugs_detect[model].append(0)
    df = pd.DataFrame()
    df['bug'] = bugs
    for model in model_bugs_detect:
        df[model] = model_bugs_detect[model]
    df.to_csv('{}.csv'.format(type),index=False)
if __name__ == '__main__':
    getStatCsv('new')
    getStatCsv('old')