# encoding=utf-8
from abc import abstractmethod, ABC
from typing import List

import pandas as pd
from joblib import delayed, Parallel


class Scorer(ABC):
    comparator = "g"

    @abstractmethod
    def score(self, result_df: pd.DataFrame):
        return


class BugFound(Scorer):
    comparator = "g"

    def score(self, result_df: pd.DataFrame):
        return (result_df.groupby(['project', 'bug_num']).TP.sum() > 0).sum()


class FPR(Scorer):
    comparator = "l"

    def score(self, result_df: pd.DataFrame):
        FPs = result_df.FP.sum()
        TNs = result_df.TN.sum()
        return FPs / (FPs + TNs)


class Precision(Scorer):
    comparator = "g"

    def score(self, result_df: pd.DataFrame):
        TPs = result_df.TP.sum()
        FPs = result_df.FP.sum()
        return TPs / (TPs + FPs)


class TPs(Scorer):
    comparator = "g"

    def score(self, result_df: pd.DataFrame):
        return result_df.TP.sum()


class FPs(Scorer):
    comparator = "l"

    def score(self, result_df: pd.DataFrame):
        return result_df.FP.sum()


class RankerScorer(Scorer, ABC):
    def score(self, ranked_failed_tests: pd.DataFrame) -> pd.Series:
        bug_df = ranked_failed_tests.groupby(["project", "bug_num"], as_index=False).apply(self.cal)

        result_line = pd.Series(dtype=float)
        for column in bug_df.columns:
            result_line[f"{column}"] = bug_df[column].sum()
        # result_df = pd.DataFrame([result_line])
        # self.logger.info(f"Result df:\n{result_df}")
        return result_line

    @abstractmethod
    def cal(self, bug_ranked_tests: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class TopkRankerScorer(RankerScorer):
    def cal(self, bug_ranked_tests: pd.DataFrame) -> pd.DataFrame:
        ks = [1, 3, 5, 10]
        result = pd.DataFrame()
        if bug_ranked_tests.TP.sum() + bug_ranked_tests.FP.sum() > 0:
            # at most one line
            for k in ks:
                result[f"top_{k}"] = [bug_ranked_tests.iloc[:k].TP.sum() > 0]
        return result


class MultiTopkScorer(Scorer):
    def __init__(self):
        super().__init__()
        self.scorer = TopkRankerScorer()

    @classmethod
    def avg_result(cls, results: List[pd.Series]) -> pd.Series:
        result_line = pd.DataFrame(results)
        result_line = result_line.mean(axis=0)
        return result_line

    def score(self, ranked_tests_list: List[pd.DataFrame]) -> pd.Series:
        tasks = [delayed(self.scorer.score)(ranked_tests) for ranked_tests in ranked_tests_list]
        results = Parallel(n_jobs=-1, prefer="processes")(tasks)
        result_line = self.avg_result(results)
        # result_df = pd.DataFrame([result_line])
        # self.logger.info(f"Result df: {result_df}")
        return result_line

