import pandas as pd
import tokenize
from io import BytesIO

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
def count_tokens(code):
    tokens = tokenizer.encode(code)
    return len(tokens)
def getfocaltest():
    df = pd.read_csv('./ori/data/fine_tune_data/assert_test_new.csv')
    source = df['source']
    target = df['target']
    f1 = open('./ori/data/raw_predictions/CodeBERT/CodeBERT_new.txt', 'r', encoding="utf-8")
    f2 = open('./ori/data/raw_predictions/CodeT5/CodeT5_new.txt', 'r', encoding="utf-8")
    f3 = open('./ori/data/raw_predictions/GraphCodeBERT/GraphCodeBERT_new.txt', 'r', encoding="utf-8")
    f4 = open('./ori/data/raw_predictions/UniXcoder/UniXcoder_new.txt', 'r', encoding="utf-8")
    f5 = open('./ori/data/raw_predictions/CodeGPT/CodeGPT_new.txt', 'r', encoding="utf-8")
    preds1 = f1.read().splitlines()
    preds2 = f2.read().splitlines()
    preds3 = f3.read().splitlines()
    preds4 = f4.read().splitlines()
    preds5 = f5.read().splitlines()
    match1 = []
    match2 = []
    match3 = []
    match4 = []
    match5 = []
    for i in range(len(preds1)):
        if preds1[i] == "match:":
            match1.append(int(preds1[i + 1]))
    for i in range(len(preds2)):
        if preds2[i] == "match:":
            match2.append(int(preds2[i + 1]))
    for i in range(len(preds3)):
        if preds3[i] == "match:":
            match3.append(int(preds3[i + 1]))
    for i in range(len(preds4)):
        if preds4[i] == "match:":
            match4.append(int(preds4[i + 1]))
    for i in range(len(preds5)):
        if preds5[i] == "match:":
            if preds5[i + 1] == 'True':
                match5.append(1)
            else:
                match5.append(0)
    # # 构建一个二维列表来统计对应的单元格

    table_data = [[[0, 0] for _ in range(6)] for _ in range(5)]

    for i in range(len(source)):
        s = source[i]
        t = target[i]
        m1 = match1[i]
        m2 = match2[i]
        m3 = match3[i]
        m4 = match4[i]
        m5 = match5[i]
        source_tokens = count_tokens(s)
        target_tokens = count_tokens(t)
        col = source_tokens // 100
        m = [m1, m2, m3, m4, m5]
        if col > 5:
            col = 5
        for j in range(5):
            table_data[j][col][0] += 1
            table_data[j][col][1] += m[j]
    for i in range(5):
        model_accu = 0
        for j in range(6):
            accu = table_data[i][j][1] / table_data[i][j][0]
            num = table_data[i][j][1]
            model_accu+=num
            table_data[i][j] = '{} ({}%)'.format(num,round(accu*100,2))
        print(model_accu)
        print(model_accu/len(source))
    # 将统计结果转换为DataFrame
    columns_labels= [f"{i * 100}-{(i + 1) * 100}" for i in range(5)]
    columns_labels.append('>500')
    index_labels = ['CodeBERT', 'CodeT5', 'GraphCodeBERT', 'UniXcoder', 'CodeGPT']
    df = pd.DataFrame(table_data, index=index_labels, columns=columns_labels)
    print(df)
    df.to_csv('statistic/focal_test.csv', encoding='utf-8')
def getassertion():
    df = pd.read_csv('./ori/data/fine_tune_data/assert_test_new.csv')
    source = df['source']
    target = df['target']
    f1 = open('./ori/data/raw_predictions/CodeBERT/CodeBERT_new.txt', 'r', encoding="utf-8")
    f2 = open('./ori/data/raw_predictions/CodeT5/CodeT5_new.txt', 'r', encoding="utf-8")
    f3 = open('./ori/data/raw_predictions/GraphCodeBERT/GraphCodeBERT_new.txt', 'r', encoding="utf-8")
    f4 = open('./ori/data/raw_predictions/UniXcoder/UniXcoder_new.txt', 'r', encoding="utf-8")
    f5 = open('./ori/data/raw_predictions/CodeGPT/CodeGPT_new.txt', 'r', encoding="utf-8")
    preds1 = f1.read().splitlines()
    preds2 = f2.read().splitlines()
    preds3 = f3.read().splitlines()
    preds4 = f4.read().splitlines()
    preds5 = f5.read().splitlines()
    match1 = []
    match2 = []
    match3 = []
    match4 = []
    match5 = []
    for i in range(len(preds1)):
        if preds1[i] == "match:":
            match1.append(int(preds1[i + 1]))
    for i in range(len(preds2)):
        if preds2[i] == "match:":
            match2.append(int(preds2[i + 1]))
    for i in range(len(preds3)):
        if preds3[i] == "match:":
            match3.append(int(preds3[i + 1]))
    for i in range(len(preds4)):
        if preds4[i] == "match:":
            match4.append(int(preds4[i + 1]))
    for i in range(len(preds5)):
        if preds5[i] == "match:":
            if preds5[i + 1] == 'True':
                match5.append(1)
            else:
                match5.append(0)
    # # 构建一个二维列表来统计对应的单元格
    table_data = [[[0, 0] for _ in range(6)] for _ in range(5)]

    for i in range(len(source)):
        s = source[i]
        t = target[i]
        m1 = match1[i]
        m2 = match2[i]
        m3 = match3[i]
        m4 = match4[i]
        m5 = match5[i]
        source_tokens = count_tokens(s)
        target_tokens = count_tokens(t)
        col = target_tokens // 20
        m = [m1, m2, m3, m4, m5]
        if col > 5:
            col = 5
        for j in range(5):
            table_data[j][col][0] += 1
            table_data[j][col][1] += m[j]
    for i in range(5):
        model_accu = 0
        for j in range(6):
            accu = table_data[i][j][1] / table_data[i][j][0]
            num = table_data[i][j][1]
            model_accu+=num
            table_data[i][j] = '{} ({}%)'.format(num,round(accu*100,2))
        print(model_accu)
        print(model_accu/len(source))
    # 将统计结果转换为DataFrame
    columns_labels= [f"{i * 20}-{(i + 1) * 20}" for i in range(5)]
    columns_labels.append('>100')
    index_labels = ['CodeBERT', 'CodeT5', 'GraphCodeBERT', 'UniXcoder', 'CodeGPT']
    df = pd.DataFrame(table_data, index=index_labels, columns=columns_labels)
    print(df)
    df.to_csv('statistic/assertion.csv', encoding='utf-8')
def getassertiontype():
    assertion_types = ['Equals','True','That','NotNull','False','Null','ArrayEquals','Same']
    assertion_types_with_assert = ['assert'+i for i in assertion_types]
    df = pd.read_csv('./ori/data/fine_tune_data/assert_test_new.csv')
    source = df['source']
    target = df['target']
    f1 = open('./ori/data/raw_predictions/CodeBERT/CodeBERT_new.txt', 'r', encoding="utf-8")
    f2 = open('./ori/data/raw_predictions/CodeT5/CodeT5_new.txt', 'r', encoding="utf-8")
    f3 = open('./ori/data/raw_predictions/GraphCodeBERT/GraphCodeBERT_new.txt', 'r', encoding="utf-8")
    f4 = open('./ori/data/raw_predictions/UniXcoder/UniXcoder_new.txt', 'r', encoding="utf-8")
    f5 = open('./ori/data/raw_predictions/CodeGPT/CodeGPT_new.txt', 'r', encoding="utf-8")
    preds1 = f1.read().splitlines()
    preds2 = f2.read().splitlines()
    preds3 = f3.read().splitlines()
    preds4 = f4.read().splitlines()
    preds5 = f5.read().splitlines()
    match1 = []
    match2 = []
    match3 = []
    match4 = []
    match5 = []
    for i in range(len(preds1)):
        if preds1[i] == "match:":
            match1.append(int(preds1[i + 1]))
    for i in range(len(preds2)):
        if preds2[i] == "match:":
            match2.append(int(preds2[i + 1]))
    for i in range(len(preds3)):
        if preds3[i] == "match:":
            match3.append(int(preds3[i + 1]))
    for i in range(len(preds4)):
        if preds4[i] == "match:":
            match4.append(int(preds4[i + 1]))
    for i in range(len(preds5)):
        if preds5[i] == "match:":
            if preds5[i + 1] == 'True':
                match5.append(1)
            else:
                match5.append(0)
    # # 构建一个二维列表来统计对应的单元格
    table_data = [[[0, 0] for _ in range(9)] for _ in range(5)]

    for i in range(len(source)):
        s = source[i]
        t = target[i]
        m1 = match1[i]
        m2 = match2[i]
        m3 = match3[i]
        m4 = match4[i]
        m5 = match5[i]
        row = 8
        n1 = t.split(" ")[0]
        n2 = t.split(" ")[2]
        if n1 in assertion_types_with_assert:
            row = assertion_types_with_assert.index(n1)
        elif n2 in assertion_types_with_assert:
            row = assertion_types_with_assert.index(n2)
        m = [m1, m2, m3, m4, m5]
        for j in range(5):
            table_data[j][row][0] += 1
            table_data[j][row][1] += m[j]
    for i in range(5):
        model_accu = 0
        for j in range(9):
            accu = table_data[i][j][1] / table_data[i][j][0]
            num = table_data[i][j][1]
            model_accu+=num
            table_data[i][j] = '{} ({}%)'.format(num,round(accu*100,2))
        print(model_accu)
        print(model_accu/len(source))
    # 将统计结果转换为DataFrame
    columns_labels= assertion_types
    columns_labels.append('Other')
    index_labels  = ['CodeBERT', 'CodeT5', 'GraphCodeBERT', 'UniXcoder', 'CodeGPT']
    df = pd.DataFrame(table_data, index=index_labels, columns=columns_labels)
    print(df)
    df.to_csv('statistic/assertiontype.csv', encoding='utf-8')
def accu(path):
    f_raw_preds = open(path, 'r', encoding="utf-8")
    preds = f_raw_preds.read().splitlines()
    preds_acc = []
    for i in range(len(preds)):
        if preds[i] == "match:":
            preds_acc.append(int(preds[i + 1]))
    print(sum(preds_acc)/len(preds_acc))
if __name__ == '__main__':
    # getfocaltest()
    # getassertion()
    # getassertiontype()
    accu('CodeT5_mask_test.txt')

