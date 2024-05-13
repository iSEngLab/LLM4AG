
import argparse, csv, os, random
import pandas as pd
import subprocess as sp
import numpy as np

import model.assertion_data as assertion_data

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
    parser.add_argument('model_name')
    args = parser.parse_args()

    base_dir = os.path.dirname(args.input_data)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    assert base_dir == os.path.dirname(args.metadata)

    fm_test_pairs = pd.read_csv(args.input_data).fillna('')
    metadata = pd.read_csv(args.metadata).fillna('')
    metadata['id'] = metadata.project + metadata.bug_num.astype(str) + metadata.test_name

    methods, tests, docstrings = fm_test_pairs.focal_method, fm_test_pairs.test_prefix, fm_test_pairs.docstring

    print('preparing assertion model inputs')
    vocab = np.load('data/evo_vocab.npy', allow_pickle=True).item()
    assert_input_file = os.path.join(base_dir, 'assert_model_inputs.csv')
    if not os.path.exists(assert_input_file):
        method_test_assert_data, idxs = assertion_data.get_model_inputs(tests, methods, vocab, metadata)
        assert_inputs_df = pd.DataFrame(method_test_assert_data,
                                        columns=["project", "bug_num", "test_name", "test_prefix", "source", "target"])
        assert_inputs_df.to_csv(assert_input_file)

    sp.run(f'bash ./model/{args.model_name}/run_eval.sh {assert_input_file}'.split(), env=os.environ.copy())

    assert_pred_file = os.path.join(base_dir, "{}_preds".format(args.model_name), "assertion_preds.csv")
    result_df = pd.read_csv(assert_pred_file)
    except_preds = [0] * len(result_df)
    result_df['except_pred'] = except_preds

    # write oracle predictions
    pred_file = os.path.join(base_dir, '{}_oracle_preds.csv'.format(args.model_name))
    result_df.to_csv(pred_file)

    print(f'wrote oracle predictions to {pred_file}')


if __name__ == '__main__':
    main()
