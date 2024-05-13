import pandas as pd
def countcsv(path):
    df = pd.read_csv(path)
    print(len(df))
if __name__ == '__main__':
    models = ['naive']
    for i in range(1, 11):
        countcsv(
            './data/evosuite_buggy_tests/{}/assert_model_inputs.csv'.format(i))
    for model in models:
        print(model)
        for i in range(1,11):
            try:
                countcsv('./data/evosuite_buggy_tests/{}/{}_generated/merged_results/failed_test_data.csv'.format(i,model))
            except Exception:
                print('error!')