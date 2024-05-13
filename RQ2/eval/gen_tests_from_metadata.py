import argparse, os, re
import abc
import logging
import sys
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from subprocess import run
from collections import defaultdict
from joblib import delayed, Parallel


fail_catch_extract_re = re.compile(r'try\s*{(.*;).*fail\(.*\)\s*;\s*}\s*catch', re.DOTALL)
assert_re = re.compile("assert\w*\s*\((.*)\)")

def get_prefix(test):
    open_curly = test.find('{')
    if not 'throws ' in test[:open_curly]:
        test = test[:open_curly] + ' throws Exception ' + test[open_curly:]

    test = test.replace('// Undeclared exception!', '')
    m_try_catch = fail_catch_extract_re.search(test)
    m_assert = assert_re.search(test)
    loc = len(test)
    if m_try_catch:
        loc = m_try_catch.span()[0]
        try_content = " " + m_try_catch.group(1).strip()

        return test[0:loc] + try_content 
    elif m_assert:
        try:
            assert m_assert #If there isn't a try catch, there should be an assertion!
        except AssertionError: 
            logging.error("no assertion or try catch in", test) 
            # sys.exit(1)
        loc = m_assert.span()[0]
        return test[0:loc] 
    else:
        return test[0:test.rfind('}')]


def bool_assert_to_equals(assertion):
    assertion = assertion.strip()
    bool_str = 'true' if 'assertTrue' in assertion else 'false'
    assert_arg = assertion[assertion.find('(')+1:assertion.rfind(')')]
    return 'assertEquals('+bool_str+', '+assert_arg+')'


def insert_assertion(method, assertion):
    lines = method.strip().split("\n")

    # if not 'assert' or 'Assert' in assertion:
    #     logging.error('ERROR invalid assertion pred:')
    #     logging.error(method)
    #     logging.error(assertion)
    #     sys.exit(0)
    #
    # if 'coreOperationGreaterThanOrEqual0' in method:
    #     logging.warning(lines)


    return "\n".join(lines + ["      " + assertion+';'] + ["}"])


def insert_try_catch(method):

    open_curly = method.find('{')
    try_catch_method = method[:open_curly] + '{\n\ttry ' + method[open_curly:] + '\n\t\tfail(\"Expecting exception\"); } catch (Exception e) { }\n\t}'

    return try_catch_method


def get_imports(java_file):
    imports = []
    with open(java_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('import'):
                imports += [line]
            if line.startswith('@RunWith'):
                break

    return imports


def gen_test_from_meta(test_harness, corpus_path, orig_corpus, project, generator, bug, test_id, full_test_name, test_case):
    full_class_name = full_test_name.split('_ESTest::')[0]
    package_name = '.'.join(full_class_name.split('.')[:-1])
    class_name = full_class_name.split('.')[-1]
    
    with open(f'{test_harness}/ESTest.java') as f:
        test_harness_template = f.read()
                
    with open(f'{test_harness}/ESTest_scaffolding.java') as f:
        scaffolding_template = f.read()

    test_case_dir = f'{corpus_path}/generated_d4j_tests/{project}/{generator}/{bug}/{test_id}/'
    orig_test_dir = f'{orig_corpus}/generated/{project}/evosuite/{bug}/'

    package_path = package_name.replace('.', '/')
    package_base_dir = package_path.split('/')[0]

    orig_test_file = orig_test_dir + package_path + f'/{class_name}_ESTest.java'
    imports = set(get_imports(orig_test_file))
    harness_imports = set(get_imports(f'{args.test_harness}/ESTest.java'))

    imports = imports - harness_imports

    class_imports = '\n'.join(imports)
    classes_str_list = ', '.join(['\"'+cls_src.replace('import ', '')+'\"' for cls_src in imports])

    filled_harness = deepcopy(test_harness_template)
    filled_harness = filled_harness.replace('{TEST_PACKAGE}', package_name)
    filled_harness = filled_harness.replace('{TEST_IMPORTS}', class_imports)
    filled_harness = filled_harness.replace('{TEST_CLASS_NAME}', class_name)
    testcase_filled_harness = filled_harness.replace('{TEST_CASES}', test_case)

    filled_scaffolding = deepcopy(scaffolding_template)
    filled_scaffolding = filled_scaffolding.replace('{TEST_PACKAGE}', package_name)
    filled_scaffolding = filled_scaffolding.replace('{TEST_CLASS}', class_name)
    filled_scaffolding = filled_scaffolding.replace('{SUPPORT_CLASSES}', classes_str_list)

    run(f'rm -r {test_case_dir}'.split(), capture_output=True)
    os.makedirs(test_case_dir + package_path)

    testcase_outfile = test_case_dir + package_path + f'/{class_name}_ESTest.java'
    with open(testcase_outfile, 'w') as f:
        f.write(testcase_filled_harness)

    with open(test_case_dir + package_path + f'/{class_name}_ESTest_scaffolding.java', 'w') as f:
        f.write(filled_scaffolding)

    with open(test_case_dir + '/test.txt', 'w') as f:
        f.write(test_case)

    cwd = os.getcwd()
    try:
        os.chdir(test_case_dir)
        run(f'tar cjf {project}-{bug}b-{generator}.{test_id}.tar.bz2 {package_base_dir}'.split())
    except Exception as e:
        logging.error('ERROR', test_case_dir)
        raise e
    finally:
        os.chdir(cwd)


class Generator(abc.ABC):
    @abc.abstractmethod
    def generate(self, test_prefix, *args, **kwargs):
        pass


class TogaGenerator(Generator):
    def generate(self, test_prefix, except_pred: bool, assert_pred: str, *args, **kwargs) -> str:
        # remove try catch and keep the statements
        # remove assert
        test = '@Test\n' + get_prefix(test_prefix)

        if except_pred:
            test = insert_try_catch(test)
        elif assert_pred:
            assertion = assert_pred
            if 'assertEquals' in test_prefix and\
                    ('assertTrue' in assertion or 'assertFalse' in assertion):
                assertion = bool_assert_to_equals(assertion)

            test = insert_assertion(test, assertion)
        else:
            test += "\n    }"
        return test


class NaiveGenerator(Generator):
    def generate(self, test_prefix, *args, **kwargs):
        test = '@Test\n' + get_prefix(test_prefix)
        test += "\n    }"
        return test


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('oracle_preds')
    parser.add_argument('original_test_corpus')
    parser.add_argument('output_dir')
    parser.add_argument('--test_harness', default='test_harness')
    parser.add_argument('--d4j_path', default='/data/swf/defects4j/framework/projects')
    parser.add_argument('-g', '--generator', dest="generator", type=str, default="toga")
    parser.add_argument('-v', action='store_true')

    args = parser.parse_args()

    test_harness = args.test_harness
    orig_corpus = args.original_test_corpus
    corpus_path = args.output_dir
    generator = None
    if args.generator == "toga":
        generator = TogaGenerator()
    elif args.generator == "naive":
        generator = NaiveGenerator()
    else:
        raise Exception(f"Unexpected generator {args.generator}: should be toga or naive") 

    metadata_df = pd.read_csv(args.oracle_preds).fillna('')

    gen_tests = []
    for row in metadata_df.itertuples():
        test = generator.generate(row.test_prefix, except_pred=row.except_pred, assert_pred=row.assert_pred)
        gen_tests += [test]

    test_ids = defaultdict(int)

    tasks = []
    meta_test_ids = []
    for idx, meta in enumerate(metadata_df.itertuples()):
        test_case = gen_tests[idx]

        bug = int(meta.bug_num)
        project = meta.project

        full_test_name = meta.test_name

        # self increment
        test_id = test_ids[project+str(bug)]
        meta_test_ids.append(test_id)
        test_ids[project+str(bug)] += 1
        
        task_arg = (test_harness, corpus_path, orig_corpus, project, args.generator, bug, test_id, full_test_name, test_case)
        tasks.append(delayed(gen_test_from_meta)(*task_arg))

    # create a csv_file to store test_ids
    metadata_df = metadata_df[["project", "bug_num"]]
    metadata_df["test_id"] = meta_test_ids

    if not args.v:
        results = Parallel(n_jobs=-1, prefer='processes')(tqdm(tasks))
    else:
        results = Parallel(n_jobs=None, prefer='processes')(tqdm(tasks))
    
    test_id_file = os.path.join(corpus_path, "test_ids.csv")
    metadata_df.to_csv(test_id_file, index=False,encoding='utf-8')
