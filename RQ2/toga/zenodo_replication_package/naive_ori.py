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

    methods, tests, docstrings = fm_test_pairs.focal_method, fm_test_pairs.test_prefix, fm_test_pairs.docstring

    # for NoException, except_preds are all 0, assert_pred are all 0, and no_except_pred are all 1
    except_preds = [0] * len(metadata)
    except_logit_0 = [1.0] * len(metadata)
    except_logit_1 = [0.0] * len(metadata)
    metadata['except_pred'] = except_preds
    metadata['except_logit_0'] = except_logit_0
    metadata['except_logit_1'] = except_logit_1

    # ASSERT INPUTS
    assert_preds = [''] * len(metadata)
    assert_logit_0 = [1.0] * len(metadata)
    assert_logit_1 = [0.0] * len(metadata)
    metadata['assert_pred'] = assert_preds
    metadata['assert_logit_0'] = assert_logit_0
    metadata['assert_logit_1'] = assert_logit_1

    # write oracle predictions
    pred_file = os.path.join(base_dir, 'naive_ori_oracle_preds.csv')

    with open(pred_file, 'w') as f:
        w = csv.writer(f)
        w.writerow(
            'project,bug_num,test_name,test_prefix,except_pred,except_logit_0,except_logit_1,assert_pred,assert_logit_0,assert_logit_1'.split(
                ','))
        for orig_test, meta in zip(tests, metadata.itertuples()):
            test_prefix = orig_test
            except_pred = meta.except_pred
            assert_pred = meta.assert_pred
            if except_pred:
                assert_pred = ''
            bug_num = meta.bug_num
            project = meta.project
            test_name = meta.test_name
            except_logit_0 = meta.except_logit_0
            except_logit_1 = meta.except_logit_1
            assert_logit_0 = meta.assert_logit_0
            assert_logit_1 = meta.assert_logit_1

            w.writerow(
                [project, bug_num, test_name, test_prefix, except_pred, except_logit_0, except_logit_1, assert_pred,
                 assert_logit_0, assert_logit_1])

    print(f'wrote oracle predictions to {pred_file}')


if __name__ == '__main__':
    main()