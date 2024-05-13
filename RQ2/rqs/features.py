# encoding=utf-8
import javalang
from joblib import delayed, Parallel
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from .common import parse_method_decl, collect_catch_exceptions
from collections import Counter, defaultdict
from typing import List, Set
from itertools import chain

FEATURE_NAMES = [
    # NTOG I/O
    'focal_name_count',
    'distinct_line',
    'except_pred',
    'no_except_pred',
    # execution output
    'test_prefix_exception',
    'trace_exception_count',
    'trace_desc_count',
    'not_rare_e',
    'rare_exception_count',
    'focal_rare_e_count',
    # Text Similarity
    'test_doc_sim',
]


def cal_test_prefix_e_name(test_prefix: str):
    tree = parse_method_decl(test_prefix)
    test_e_names = set(e.split('.')[-1] for e in collect_catch_exceptions(tree))
    return test_e_names


def extract_method_name(method: str):
    if not method:
        return ''
    try:
        tokens = javalang.tokenizer.tokenize(method)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
    except Exception as e:
        # print(method)
        return ''
    return tree.name


def extract_focal_name(method: str):
    return extract_method_name(method)


def extract_trace_exception(trace: str):
    return trace.strip().split('\n')[1].strip().split(":")[0].strip()


def extract_trace_exception_name(trace_exception: str):
    trace_e_name = trace_exception.split('.')[-1]
    if trace_exception.startswith("org.evosuite") and trace_e_name.startswith("Mock"):
        trace_e_name = trace_e_name[4:]
    return trace_e_name


def extract_trace_desc(trace: str):
    items = trace.strip().split('\n')[1].strip().split(":")
    return ":".join(items[1:]).strip()


def extract_trace_exception_has_desc(trace_desc: str):
    return True if trace_desc else False


def cal_exception_type(e_name: str, trace_exception: str, project: str):
    if trace_exception.startswith("java."):
        return "java"
    elif project.lower() in ''.join(trace_exception.split('.')):
        return "project"
    elif trace_exception.startswith("org.evosuite"):
        return "evosuite"
    elif e_name == "AssertionFailedError":
        return "assert"
    else:
        return "other"


def build_vectorizer(df: pd.DataFrame, use_trace: bool = False):
    focal_methods = set(df["focal_method"])
    docstrings = set(df["docstring"])
    test_prefixes = list(df["test_prefix"])
    generated_tests = list(df["generated_test"])
    trace_lines = [trace.strip().split('\n')[1] for trace in df["buggy_test_trace"]]

    vectorizer = TfidfVectorizer(stop_words="english")
    if use_trace:
        corpus = list(focal_methods) + list(docstrings) + test_prefixes + generated_tests + trace_lines
    else:
        corpus = list(focal_methods) + list(docstrings) + test_prefixes + generated_tests
    vectorizer.fit(corpus)
    return vectorizer


def cal_sim(vectorizer, left: str, right: str):
    vectors = vectorizer.transform([left, right])
    # print(vectors.todense())
    sim = vectors[0].dot(vectors[1].T)[0, 0]
    return sim


def collect_test_lines(method: str):
    # remove the first two and the last one
    lines = [line.strip() for line in method.strip().split('\n') if line.strip()]
    return set(lines[2:-1])


def cal_distinct_strs(strs: List[Set]):
    results = []
    pre_invos = set()
    for idx, invo in enumerate(strs):
        suc_invos = set(chain(*strs[idx + 1:]))
        distinct_invos = (invo - pre_invos) - suc_invos
        results.append(distinct_invos)
        pre_invos.update(invo)
    return results


class FeatureExtractor:
    def __init__(self, df: pd.DataFrame, trace_in_corpus: bool = False):
        self.df = df
        self.vectorizer = build_vectorizer(self.df, use_trace=trace_in_corpus)

    def extract(self, feature_list: List[str]):
        # store the old df
        old_columns = list(self.df.columns)
        # force to reset some columns
        self.df['assert_pred'] = self._assert_pred()
        self.df['except_pred'] = self._except_pred()
        for feature in feature_list:
            self.df[feature] = getattr(self, f"_{feature}")()
        feature_list = list(set(feature_list) - set(old_columns))
        return self.df[old_columns + feature_list]

    def extract_if_not_exist(self, name: str):
        if name not in self.df.columns:
            self.df[name] = getattr(self, f"_{name}")()

    def cal_sim_delta(self, first: str, second: str):
        self.extract_if_not_exist(first)
        self.extract_if_not_exist(second)
        return self.df[first] - self.df[second]

    def _assert_pred(self):
        return (self.df['assert_pred'].str.strip() != "").astype(bool)

    def _except_pred(self):
        return self.df['except_pred'].astype(bool)

    def _no_except_pred(self):
        self.extract_if_not_exist("assert_pred")
        self.extract_if_not_exist("except_pred")
        assert_pred = self.df['assert_pred']
        except_pred = self.df['except_pred']
        if assert_pred.dtype == str or except_pred.dtype == str:
            print(self.df)
        return ~assert_pred & ~except_pred

    def _prefix_e_names(self):
        return self.df['test_prefix'].transform(cal_test_prefix_e_name)

    def _test_prefix_exception(self):
        self.extract_if_not_exist("prefix_e_names")
        return self.df['prefix_e_names'].transform(lambda x: len(x) > 0)

    def _trace_exception_name(self):
        self.df['trace_exception'] = self.df['buggy_test_trace'].transform(extract_trace_exception)
        return self.df['trace_exception'].transform(extract_trace_exception_name)

    def _trace_exception_count(self):
        self.extract_if_not_exist("trace_exception_name")
        return self.df['trace_exception'].map(self.df['trace_exception'].value_counts())

    def _trace_exception_has_desc(self):
        self.df['trace_desc'] = self.df['buggy_test_trace'].transform(extract_trace_desc)
        return self.df['trace_desc'].transform(extract_trace_exception_has_desc)

    def _trace_desc_count(self):
        self.extract_if_not_exist("trace_exception_has_desc")
        return self.df['trace_desc'].map(self.df['trace_desc'].value_counts())

    def _trace_exception_type(self):
        self.extract_if_not_exist("trace_exception_name")
        trace_exception_types = []
        for row in self.df.itertuples():
            trace_exception_types.append(cal_exception_type(row.trace_exception_name, row.trace_exception, row.project))
        return trace_exception_types

    def _trace_exception_in_focal(self):
        self.extract_if_not_exist("trace_exception_name")
        results = []
        for row in self.df.itertuples():
            results.append(row.trace_exception_name in row.focal_method)
        return results

    def _trace_exception_in_doc(self):
        self.extract_if_not_exist("trace_exception_name")
        results = []
        for row in self.df.itertuples():
            results.append(row.trace_exception_name in row.docstring)
        return results

    def _not_rare_e(self):
        self.extract_if_not_exist("trace_exception_type")
        self.extract_if_not_exist("trace_exception_in_focal")
        self.extract_if_not_exist("trace_exception_in_doc")
        return [(row.trace_exception_type == "assert") | row.trace_exception_in_focal | row.trace_exception_in_doc
                for row in self.df.itertuples()]

    def _rare_exception_count(self):
        self.extract_if_not_exist('not_rare_e')
        self.extract_if_not_exist('trace_exception_name')

        rare_exception_names = []
        for row in self.df.itertuples():
            if not row.not_rare_e:
                rare_exception_names.append(row.trace_exception_name)
        rare_exception_counter = Counter(rare_exception_names)
        results = []
        for row in self.df.itertuples():
            if row.not_rare_e:
                results.append(0)
            else:
                results.append(rare_exception_counter[row.trace_exception_name])
        return results

    def _focal_rare_e_count(self):
        self.extract_if_not_exist('not_rare_e')
        self.extract_if_not_exist("trace_exception_name")
        focal_rare_e_counter = defaultdict(lambda: 0)
        for row in self.df.itertuples():
            if not row.not_rare_e:
                focal_rare_e_counter[(row.focal_method, row.trace_exception_name)] += 1
        focal_rare_e_counts = [focal_rare_e_counter[(row.focal_method, row.trace_exception_name)]
                               for row in self.df.itertuples()]
        return focal_rare_e_counts

    def _test_doc_sim(self):
        return [cal_sim(self.vectorizer, row.generated_test, row.docstring) for row in self.df.itertuples()]

    def _focal_name(self):
        return self.df['focal_method'].transform(extract_focal_name)

    def _focal_name_count(self):
        self.extract_if_not_exist("focal_name")
        return self.df['focal_name'].map(self.df['focal_name'].value_counts())

    def _distinct_line(self):
        lines_list = self.df['generated_test'].transform(collect_test_lines)
        return [len(lines) for lines in cal_distinct_strs(lines_list)]


def cal_features(failed_tests: pd.DataFrame, n_jobs: int, feature_list: List[str] = None,
                 trace_in_corpus: bool = False):
    if feature_list is None:
        feature_list = FEATURE_NAMES
    tasks = []
    for group_name, bug in failed_tests.groupby(["project", "bug_num"], as_index=False):
        extractor = FeatureExtractor(bug, trace_in_corpus=trace_in_corpus)
        tasks.append(delayed(extractor.extract)(feature_list))
    results = Parallel(n_jobs=n_jobs, prefer="processes")(tasks)
    new_failed_tests = pd.concat(results, axis=0)
    return new_failed_tests


def cal_features_raw(failed_tests: pd.DataFrame, feature_list: List[str] = None,
                     trace_in_corpus: bool = False):
    if feature_list is None:
        feature_list = FEATURE_NAMES
    results = []
    for group_name, bug in failed_tests.groupby(["project", "bug_num"], as_index=False):
        extractor = FeatureExtractor(bug, trace_in_corpus=trace_in_corpus)
        results.append(extractor.extract(feature_list))
    new_failed_tests = pd.concat(results, axis=0)
    return new_failed_tests


def construct_test_features_raw(bug_failed_tests: pd.DataFrame, feature_list: List[str]):
    for feature in feature_list:
        assert (not feature.endswith("_type") and not feature.endswith("_name"))
    X = bug_failed_tests[feature_list]
    return X
