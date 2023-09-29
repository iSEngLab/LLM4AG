# encoding=utf-8

import os
from functools import partial
from typing import List, Union

import fire
import pandas as pd
from abc import ABC, abstractmethod

from joblib import delayed, Parallel
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

from rqs.features import cal_features, construct_test_features_raw, FEATURE_NAMES, cal_features_raw
from rqs.metrics import MultiTopkScorer, TopkRankerScorer
from rqs.wilcoxon import wilcoxon_test


class Step(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        pass


class FailedTestLoader(Step):
    def run(self, failed_test_file: str) -> pd.DataFrame:
        failed_test_file = failed_test_file
        failed_tests = pd.read_csv(failed_test_file).fillna("")
        length1 = len(failed_tests)
        failed_tests = failed_tests.drop_duplicates(
            subset=["project", "bug_num", "focal_method", "docstring", "generated_test"])
        # we already drop duplicates
        assert length1 == len(failed_tests)
        return failed_tests


class FeatureBuilder(Step):
    def run(self, failed_tests: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
        return cal_features(failed_tests, n_jobs)


class FeatureLoader(Step):
    def run(self, result_dir: str) -> pd.DataFrame:
        feature_file = os.path.join(result_dir, "failed_test_features.csv")
        return pd.read_csv(feature_file).fillna("")


class FullFeatureBuilderRaw(Step):
    def run(self, failed_tests: pd.DataFrame) -> pd.DataFrame:
        return cal_features_raw(failed_tests)


class Ranker(Step, ABC):
    def run(self, failed_tests: pd.DataFrame) -> pd.DataFrame:
        grouped_bugs = failed_tests.groupby(["project", "bug_num"], as_index=False)
        ranked_failed_tests = grouped_bugs.apply(self.rank)
        return ranked_failed_tests

    @abstractmethod
    def rank(self, bug_failed_tests: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class RandomRanker(Ranker):
    def __init__(self, random_state: int, **kwargs):
        super().__init__(**kwargs)
        self.random_state = random_state

    def rank(self, bug_failed_tests: pd.DataFrame) -> pd.DataFrame:
        return bug_failed_tests.sample(frac=1, random_state=self.random_state)


class MultiRandomRanker(Step):
    def __init__(self, random_states: List[int], **kwargs):
        super().__init__(**kwargs)
        self.rankers = [RandomRanker(state) for state in random_states]

    def run(self, failed_tests: pd.DataFrame, n_jobs: int = -1) -> List[pd.DataFrame]:
        tasks = []
        for ranker in self.rankers:
            tasks.append(delayed(ranker.run)(failed_tests))
        result = Parallel(n_jobs=n_jobs, prefer="processes")(tasks)
        return result


def find_isof_outliers(X: pd.DataFrame, random_state: int, contamination: Union[str, float] = "auto"):
    clf = IsolationForest(random_state=random_state, contamination=contamination)
    scores = clf.fit(X).decision_function(X)
    # small one is outlier
    # print(scores)
    return scores


class ClusterRanker(Step, ABC):
    def run(self, failed_tests: pd.DataFrame, feature_list: List[str] = None, n_jobs: int = -1) -> pd.DataFrame:
        tasks = []
        for group_name, bug in failed_tests.groupby(["project", "bug_num"], as_index=False):
            rank_func = partial(self.rank, feature_list=feature_list)
            tasks.append(delayed(rank_func)(bug))
        results = Parallel(n_jobs=n_jobs, prefer="processes")(tasks)
        ranked_failed_tests = pd.concat(results, axis=0)
        return ranked_failed_tests

    @abstractmethod
    def rank(self, bug_failed_tests: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        raise NotImplementedError


class IsofClusterRanker(ClusterRanker):
    def __init__(self, random_state: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.random_state = random_state

    def rank(self, bug_failed_tests: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        if feature_list is None:
            features = FEATURE_NAMES
        else:
            features = feature_list
        X = construct_test_features_raw(bug_failed_tests, features)
        scores = self.find_outliers(X)
        bug_failed_tests['cls_score'] = scores
        bug_failed_tests = bug_failed_tests.sort_values(by=["cls_score"], ascending=True)
        bug_failed_tests = bug_failed_tests.drop('cls_score', axis=1)
        return bug_failed_tests

    def find_outliers(self, X):
        return find_isof_outliers(X, self.random_state)


class MultiIsofClusterRanker(Step):
    def __init__(self, random_states: List[int], **kwargs):
        super().__init__(**kwargs)
        self.rankers = [IsofClusterRanker(random_state=state) for state in random_states]

    def run(self, failed_tests: pd.DataFrame, feature_list: List[str] = None, n_jobs: int = -1) -> List[pd.DataFrame]:
        result = []
        for ranker in self.rankers:
            result.append(ranker.run(failed_tests, feature_list, n_jobs))
        return result


def dump_features(data_dir="data"):
    loader = FailedTestLoader()
    feature_builder = FeatureBuilder()
    for method in ["naive", "toga"]:
        print(method)
        for i in tqdm(range(1, 11)):
            result_dir = os.path.join(data_dir, f"evosuite_buggy_tests/{i}/{method}_generated/merged_results")
            failed_test_file = os.path.join(result_dir, "failed_test_data.csv")
            failed_tests = loader.run(failed_test_file)
            failed_test_features = feature_builder.run(failed_tests)
            out_file = os.path.join(result_dir, "failed_test_features.csv")
            failed_test_features.to_csv(out_file)


def cal_result(data_dir="data"):
    loader = FailedTestLoader()
    feature_loader = FeatureLoader()

    scorers = {
        "Multi": MultiTopkScorer(),
    }
    ranker_classes = [MultiRandomRanker, MultiIsofClusterRanker]
    df_lines = []
    indexes = []

    for method in ["naive", "toga"]:
        for ranker_class in ranker_classes:
            results = []
            print(method)
            print(ranker_class.__name__)
            for i in tqdm(range(1, 11)):
                ranker = ranker_class(list(range(0, 10)))

                failed_test_file = os.path.join(data_dir,
                                    f"evosuite_buggy_tests/{i}/{method}_generated/merged_results/failed_test_data.csv")
                failed_tests = loader.run(failed_test_file)
                if ranker_class.__name__.endswith("RandomRanker"):
                    failed_test_features = failed_tests
                else:
                    result_dir = os.path.dirname(failed_test_file)
                    failed_test_features = feature_loader.run(result_dir)

                ranked_failed_test = ranker.run(failed_test_features)
                lines = []
                for name, scorer in scorers.items():
                    inner_result_line = scorer.score(ranked_failed_test)
                    inner_result_line = inner_result_line.add_prefix(name)
                    lines.append(inner_result_line)
                inner_result = pd.concat(lines, axis=0)
                results.append(inner_result)
            result_df = pd.DataFrame(results)
            dump_df = result_df[['Multitop_1','Multitop_3', 'Multitop_5', 'Multitop_10']]
            dump_df.to_csv(f"results/ranking_df_{method}_{ranker_class}.csv")
            result_line = result_df.mean()
            df_lines.append(result_line)
            indexes.append(f"{method}_{ranker_class.__name__}")
    final_df = pd.DataFrame(df_lines, index=indexes)
    print(final_df)
    final_df.to_csv(f"results/ranking_results.csv")


if __name__ == "__main__":
    fire.Fire({
        "dump_features": dump_features,
        "cal_result": cal_result,
    })
