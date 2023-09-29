import pandas as pd
import json


def getIRdata(type):
    # 打开 JSONL 文件
    searched_assertion = []
    with open('OldDataset/{}.jsonl'.format(type), 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每一行的 JSON 对象
            data = json.loads(line)
            # 现在可以使用 data 变量来访问每个 JSON 对象的内容
            searched_assertion.append(data['src_desc'])
    df = pd.read_csv('assert_{}_old.csv'.format(type))
    source = df['source']
    add_source = []
    assert len(source) == len(searched_assertion)
    for s, ss in zip(source,searched_assertion):
        search_string = '"<AssertPlaceHolder>" ;'
        index = s.find(search_string)
        modified_string = s[:index + len(search_string)] + ' /* ' + ss + ' */ '+ s[
                                                                       index + len(
                                                                           search_string):]

        add_source.append(modified_string)
    df['source'] = add_source
    df.to_csv('assert_{}_old_IR.csv'.format(type),index=False)
def split(data):
    df = pd.read_csv(data)
    leng = int(len(df)/4)
    df[0:leng].to_csv('assert_test_old_IR_0.csv')
    df[leng:2*leng].to_csv('assert_test_old_IR_1.csv')
    df[2*leng:3*leng].to_csv('assert_test_old_IR_2.csv')
    df[3*leng:].to_csv('assert_test_old_IR_3.csv')
def combine():
    f = open('CodeT5_oldIR.txt','a')
    for i in range(4):
        f1 = open('CodeT5_None_{}.txt'.format(i),'r')
        lines = f1.read().splitlines()
        for l in lines:
            f.write(l+'\n')
        f1.close()
    f.close()
    df = pd.DataFrame()
    for i in range(4):
        df1 = pd.read_csv('CodeT5_raw_preds_{}.csv'.format(i))
        df = df.append(df1, ignore_index=True)
    df.to_csv('CodeT5_raw_preds.csv')
    accuracy = df["correctly_predicted"]
    print(sum(accuracy)/len(accuracy))
if __name__ == '__main__':
    # getIRdata('test')
    # getIRdata('val')
    # getIRdata('train')
    split('assert_test_old_IR.csv')
