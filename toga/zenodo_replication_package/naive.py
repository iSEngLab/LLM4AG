# encoding=utf-8

"""
Create naive_oracle_preds.csv for latter process
"""

import argparse, csv, os, random
import pandas as pd

pd.options.mode.chained_assignment = None

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def main():
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('input_data')
    parser.add_argument('metadata')
    parser.add_argument('--dry', action="store_true")
    args = parser.parse_args()

    base_dir = os.path.dirname(args.input_data)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    assert base_dir == os.path.dirname(args.metadata)

    fm_test_pairs = pd.read_csv(args.input_data).fillna('')
    metadata = pd.read_csv(args.metadata).fillna('')
    metadata['id'] = metadata.project + metadata.bug_num.astype(str) + metadata.test_name

    assert_pred_file = os.path.join(base_dir, "CodeBERT_preds", "assertion_preds.csv")
    result_df = pd.read_csv(assert_pred_file)
    except_preds = [0] * len(result_df)
    result_df['except_pred'] = except_preds

    # write oracle predictions
    pred_file = os.path.join(base_dir, 'naive_oracle_preds.csv')
    result_df.to_csv(pred_file)
    print(f'wrote oracle predictions to {pred_file}')


if __name__ == '__main__':
    main()
