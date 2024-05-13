# encoding=utf-8

"""
Utils for defects4j
"""

from typing import List


D4J_PROJS = ["Chart", "Cli", "Closure", "Codec", "Collections", "Compress", "Csv", "Gson", "JacksonCore",
             "JacksonDatabind", "JacksonXml", "Jsoup", "JxPath", "Lang", "Math", "Mockito", "Time"]


def parse_projects(proj: str = None) -> List[str]:
    if proj is None:
        projs = D4J_PROJS
    elif isinstance(proj, str):
        projs = proj.split(',')
    elif isinstance(proj, tuple) or isinstance(proj, list):
        projs = proj
    else:
        raise ValueError("Projects {} is unknown".format(proj))

    return projs
