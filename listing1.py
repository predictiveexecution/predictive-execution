import ast
import gast
import gast.gast as gg
import python_graphs
from python_graphs import program_graph, control_flow, program_utils
from python_graphs import program_graph_dataclasses as pb
import listing2.py

TEST_PROGRAM1 = """a = 10
b = 5
print(b - a)
"""

TEST_PROGRAM2 = """a = 10
b = 5
if a < b:
    print(b - a)
else:
    print(a - b)
"""

TEST_PROGRAM3 = """def test_function():
    a = 10
    b = 5
    return a + b

test_function()
"""

TEST_PROGRAM4 = """def test_function():
    a = 10
    b = 5
    if a < b:
        return b - a
    else:
        return a - b

test_function()
"""

TEST_PROGRAM5 = """def test_function1(a, b):
    return a + b

def test_function2():
    a = 10
    b = 5
    c = test_function1(a, b)
    return c

test_function2()
"""

# Check https://github.com/serge-sans-paille/gast/blob/master/gast/gast.py
# for complete list of AST Statment nodes.
SIMPLE_STMT_NODES = [
    gg.Delete, gg.Assign, gg.AugAssign, gg.AnnAssign, gg.Print,
    gg.Raise, gg.Assert, gg.Import, gg.ImportFrom, gg.Exec,
    gg.Global, gg.Nonlocal, gg.Expr,
    # Jump Statement nodes
    gg.Pass, gg.Break, gg.Return, gg.Continue,
]

# if_stmt, for_stmt, while_stmt, match_stmt
COMPOUND_BRANCH_STMT_NODES = [
    gg.If, gg.For, gg.AsyncFor, gg.While, gg.Match,
]

# with_stmt, try_stmt, function_def, class_def
COMPOUND_NON_BRANCH_STMT_NODES = [
    gg.With, gg.AsyncWith, gg.Try, gg.TryStar, gg.FunctionDef,
    gg.AsyncFunctionDef, gg.ClassDef,
]


def get_ast_and_control_flow_graph(code):
  control_flow_visitor = control_flow.ControlFlowVisitor()
  tree = program_utils.program_to_ast(code)
  control_flow_visitor.run(tree)
  return tree, control_flow_visitor.graph


def get_lineno(node):
    if hasattr(node, 'lineno'):
        return node.lineno - 1
    else:
        if isinstance(node, gg.arguments):
            all_linenos = [arg.lineno for arg in node.posonlyargs + node.args \
                            if hasattr(arg, 'lineno')]
            return min(all_linenos) - 1
        else: return -1

def extract_intrinsic_call_edges(tree):
    definitions, calls = {}, []
    for node in ast.walk(tree):
        if isinstance(node, gg.FunctionDef):
            return_node = None
            for item in node.body:
                if isinstance(item, gg.Return):
                    return_node = item
                    break
            return_lineno = None
            if return_node: return_lineno = return_node.lineno - 1
            definitions[str(node.name)] = {
                'defn_lineno': node.lineno - 1,
                'defn_node': node,
                'return_lineno': return_lineno,
                'return_node': return_node,
            }
        elif isinstance(node, gg.Call):
            if isinstance(node.func, gg.Name):
                calls.append((str(node.func.id), node.lineno - 1, node))
            elif isinstance(node.func, gg.Attribute):
                calls.append((str(node.func.value), node.lineno - 1, node))

    call_edges = {'call': [], 'return': []}
    for item in calls:
        if item[0] not in definitions: continue
        to_item = definitions[item[0]]
        call_edges['call'].append({
            'call_node': item[2],
            'call_lineno': item[1],
            'to_node': to_item['defn_node'],
            'to_lineno': to_item['defn_lineno'],
        })
        if to_item['return_node']:
            call_edges['return'].append({
                'return_node': to_item['return_node'],
                'return_lineno': to_item['return_lineno'],
                'to_node': item[2],
                'to_lineno': item[1],
            })
    return call_edges

def extract_pdg_edges(full_graph):
    pdg_edges = {}
    for edge in full_graph.edges:
        if edge.type in [pb.EdgeType.LAST_READ, pb.EdgeType.LAST_WRITE]:
            from_node = full_graph.get_node(edge.id1).ast_node
            to_node = full_graph.get_node(edge.id2).ast_node
            from_line, to_line = get_lineno(from_node), get_lineno(to_node)
            try:
                var_from = full_graph.get_node(edge.id1).ast_node.id
                var_to = full_graph.get_node(edge.id2).ast_node.id
            except AttributeError: continue
            if var_from != var_to: continue
            if from_line == to_line: continue
            if from_line in pdg_edges:
                pdg_edges[from_line] += [(to_line, edge)]

    pdg_edges = dict(zip(pdg_edges.keys(),
                         [sorted(to_lines, key=lambda x:x[0]) for to_lines in pdg_edges.values()]))
    return pdg_edges

def get_last_line(cfg):
    max = -1
    for node in cfg.nodes:
        if hasattr(node.instruction.node, 'lineno'):
            if node.instruction.node.lineno > max:
                max = node.instruction.node.lineno - 1
    return max

def get_next_in_control_flow_order(current, cfg, call_edges):
    if isinstance(current.instruction.node, gg.Return):
        return get_target_function(current, cfg, call_edges, key='return')
    elif isinstance(current.instruction.node, gg.Expr) and \
         isinstance(current.instruction.node.value, gg.Call):
        return get_target_function(current, cfg, call_edges, key='call')
    else:
        next = list(current.next)
        return next[0]

def get_target_function(current, cfg, call_edges, key):
    if key == 'call': current_node = current.instruction.node.value
    elif key == 'return': current_node = current.instruction.node
    next = None
    for call in call_edges[key]:
        from_node, from_lineno = call[f'{key}_node'], call[f'{key}_lineno']
        if from_node == current_node:
            next = cfg.get_control_flow_node_by_ast_node(call['to_node'])
            print(next.instruction.node.lineno)
            break
    return next


def run_predictive_execution(code):
    tree, cfg = get_ast_and_control_flow_graph(code)
    pg = program_graph.get_program_graph(code)
    execution_trace = []
    call_edges = extract_intrinsic_call_edges(tree)
    pdg_edges = extract_pdg_edges(pg)
    current = cfg.get_start_control_flow_node()
    current_lineno = get_lineno(current.instruction.node)
    eof = get_last_line(cfg)

    while (current_lineno != eof):
        current_node_type = type(current.instruction.node)
        execution_trace.append(current_lineno)
        next = None
        if isinstance(current_node_type, gg.Expr) and \
           isinstance(current.instruction.node.value, gg.Call):
           next = get_target_function(current, cfg, call_edges, 'call')
        else:
            if current_node_type in SIMPLE_STMT_NODES:
                next = get_next_in_control_flow_order(current, cfg, call_edges)
            else:
                if current_node_type in COMPOUND_BRANCH_STMT_NODES:
                    next = listing2.next_statement_predictor(cfg, pg, execution_trace, current, current_lineno)
                    pass
                elif current_node_type in COMPOUND_NON_BRANCH_STMT_NODES:
                    next = get_next_in_control_flow_order(current, cfg, call_edges)
        current = next
        if(next is None):
            break
        current_lineno = get_lineno(current.instruction.node)
    execution_trace.append(eof)

if __name__ == "__main__":
    print(run_predictive_execution(TEST_PROGRAM2))