# encoding=utf-8
import os
import re
import argparse
import pandas as pd
import tqdm
from .gen_tests_from_metadata import TogaGenerator, NaiveGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='input_dir')
    parser.add_argument(dest='result_dir')
    parser.add_argument(dest='model_name')
    args = parser.parse_args()

    gen_dir = args.input_dir
    result_dir = args.result_dir
    work_dir = os.path.join(gen_dir, "..")

    pattern = re.compile(r'.*/(.*?)_generated.*')
    match = pattern.match(gen_dir)
    assert match
    method = match.group(1)
    print(f"System: {method}")
    generator = TogaGenerator() if method != "naive" else NaiveGenerator()
    oracle_pred_file = "{}_oracle_preds.csv".format(args.model_name) if method != "naive" else "naive_oracle_preds.csv"

    inputs = pd.read_csv(os.path.join(work_dir, "assert_model_inputs.csv")).fillna("")
    metas = pd.read_csv(os.path.join(work_dir, "assert_model_inputs.csv")).fillna("")
    oracle_preds = pd.read_csv(os.path.join(work_dir, oracle_pred_file)).fillna("")
    oracle_preds = oracle_preds[
        "test_prefix,except_pred,assert_pred".split(',')]
    assert len(inputs) == len(metas)

    # merge the idx,test id and unique_test_name
    uniq_test_names = pd.read_csv(os.path.join(gen_dir, "uniq_test_names.csv"))
    assert not uniq_test_names.isna().any().any()
    name_to_input = pd.concat([uniq_test_names, oracle_preds], axis=1)

    # do sanity check
    for mrow, urow in zip(metas[["project", "bug_num"]].itertuples(),
                          uniq_test_names[["project", "bug_num"]].itertuples()):
        assert mrow == urow

    # calculate the interaction between failed_tests
    test_result_df = pd.read_csv(os.path.join(gen_dir, result_dir, "test_data.csv"))
    test_result_df = test_result_df.rename(columns={"test_name": "unique_test_name"})
    print(name_to_input.columns.tolist())
    print(test_result_df.columns.tolist())
    full_test_data = name_to_input.merge(test_result_df, how="outer", on=["project", "bug_num", "unique_test_name"],
                                         validate="one_to_one")
    gen_tests = []

    # re-generate test based on test_prefix and the oracle preds
    origin_gen_dir = os.path.join(gen_dir, "generated_d4j_tests")
    print(full_test_data.columns.tolist())
    for row in tqdm.tqdm(full_test_data.itertuples()):
        if type(row.test_prefix) == float:
            test_case = ""
        else:
            test_case = generator.generate(row.test_prefix, except_pred=row.except_pred, assert_pred=row.assert_pred)
        gen_tests.append(test_case)
    full_test_data.insert(7, "generated_test", gen_tests)
    old_length = len(full_test_data)
    dup_keys = ["project", "bug_num", "generated_test"]
    full_test_data = full_test_data.drop_duplicates(subset=dup_keys)
    drop_length = old_length - len(full_test_data)
    print(f"Drop {drop_length} duplicates")
    full_test_data.to_csv(os.path.join(gen_dir, result_dir, "full_test_data.csv"), index=False,encoding='utf-8' )

    failed_test_data = full_test_data.loc[full_test_data['failed_buggy'] == True]
    assert not failed_test_data.isna().any().any()
    failed_test_data.to_csv(os.path.join(gen_dir, result_dir, "failed_test_data.csv"), index=False,encoding='utf-8')
