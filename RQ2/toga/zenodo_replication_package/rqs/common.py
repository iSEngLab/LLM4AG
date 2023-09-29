# encoding=utf-8
import javalang


def parse_method_decl(code: str):
    tokens = javalang.tokenizer.tokenize(code)
    parser = javalang.parser.Parser(tokens)
    return parser.parse_member_declaration()


def collect_catch_exceptions(tree: javalang.tree.Node):
    exceptions = []
    for path, node in tree.filter(javalang.tree.CatchClauseParameter):
        exceptions += node.types
    return exceptions


def collect_invocations(test_case: str):
    test_case = test_case
    tokens = javalang.tokenizer.tokenize(test_case)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    assert type(tree) == javalang.tree.MethodDeclaration
    invocations = set()
    for path, node in tree.filter(javalang.tree.MethodInvocation):
        invocations.add(node.member)
    for path, node in tree.filter(javalang.tree.ClassCreator):
        # we add all information of a type
        invocations.add(str(node.type))
    return invocations
