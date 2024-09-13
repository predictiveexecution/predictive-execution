import ast
import gast
import gast.gast as gg
import python_graphs
from python_graphs import program_graph, control_flow, program_utils
from python_graphs import program_graph_dataclasses as pb
import listing2
import time
import helper


TEST_PROGRAM1 = """s = 'RUDLUDR'
import sys
for i in range(len(s)):
    if (i + 1) % 2 == 1 and s[i] not in 'RUD':
        print('No')
        sys.exit(1)
    if (i + 1) % 2 == 0 and s[i] not in 'LUD':
        print('No')
        sys.exit(1)
print('Yes')
"""

TEST_PROGRAM2 = """a = 1
b = 1
if a == b: 
    b = 0
else:
    a = 0
a = 10
b = 5
if a > b:
    print(a - b)
if a < b:
    print(b - a)
elif a != b:
    a = b
    a = 1/(a - b)
else:
    print(a - b)
"""

TEST_PROGRAM3 = """import math
n = 9
while n > 0:
    n = n-5
    sqrt(n)
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

TEST_PROGRAM5 = """import sys
count = 0
for s in "123 123":
    L=[]
    for d in s.strip().split():
        count+=1
        print(f"loop count{count}")
        if d=="-":
            d=(-(L.pop()-L.pop()))
        elif d=="+":
            d=(L.pop()+L.pop())
        elif d=="*":
            d=(L.pop()*L.pop())
        elif d=="/":
            d=(1/L.pop()*L.pop())
        L.append(float(d))
    print ("%.6f"%L[0])
while count< len(input):
    count+=1
    print(count)
"""

# Check https://github.com/serge-sans-paille/gast/blob/master/gast/gast.py
# for complete list of AST Statment nodes.
SIMPLE_STMT_NODES = [
    gg.Delete, gg.Assign, gg.AugAssign, gg.AnnAssign, gg.Print,
    gg.Raise, gg.Assert, gg.Import, gg.ImportFrom, gg.Exec,
    gg.Global, gg.Nonlocal, gg.Expr,
    # Jump Statement nodes
    gg.Pass, gg.Break, gg.Return, gg.Continue, gg.Import
]

# if_stmt, for_stmt, while_stmt, match_stmt
COMPOUND_BRANCH_STMT_NODES = [
    gg.If, gg.For, gg.AsyncFor, gg.While, gg.Match, gg.Compare
]
"""
New Python_Graphs no longer accepting gg.Match for PDG generation, Please avoid testcases with match statement.
Added gg.Compare
"""

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


### need change this function, it is not going back to the loop
def get_next_in_control_flow_order(current, cfg, call_edges):
    if isinstance(current.instruction.node, gg.Return):
        return get_target_function(current, cfg, call_edges, key='return')
    elif isinstance(current.instruction.node, gg.Expr) and \
         isinstance(current.instruction.node.value, gg.Call):
        return get_target_function(current, cfg, call_edges, key='call')
    else:
        next = list(current.next)
        return next[0]
### maybe need change this function.
### any call or
def get_target_function(current, cfg, call_edges, key):
    if key == 'call': current_node = current.instruction.node.value
    elif key == 'return': current_node = current.instruction.node
    next = None
    for call in call_edges[key]:
        from_node, from_lineno = call[f'{key}_node'], call[f'{key}_lineno']
        if from_node == current_node:
            next = cfg.get_control_flow_node_by_ast_node(call['to_node'])
            # print(next.instruction.node.lineno)
            break
    return next




def traverse_cfg_with_dfs(node, tree, visited = None):
    """
    Traverse the entire CFG.
    Parameters:
        -- node: CFG node, Expecting the starting CFG node.
        -- tree: AST tree from GAST.
        -- visited: Holder for visited nodes call back.
    """
    if visited is None:
        visited = set()
    ## Point for this function is to check if my searching algorithm is working properly, if yes
    ## it should print out AST node first, then print out the corresponding CFG node, 
    ## Their Line Number supposed to be matched.
    visited.add(node)
    print(f"this is AST node {helper.find_node_by_line(tree, node.instruction.node.lineno)}")
    print(f"this is CFG node {node.instruction.node}")
    for neighbor in node.next:
        if neighbor not in visited:
            traverse_cfg_with_dfs(neighbor,tree, visited)

def run_predictive_execution(code):
    tree, cfg = get_ast_and_control_flow_graph(code)
    execution_trace = []
    call_edges = extract_intrinsic_call_edges(tree)
    current = cfg.get_start_control_flow_node()

    # while True: 
    #     current = list(current.next)[0]
    #     print(current.instruction.node.lineno)
    # exit()
    # exit()
    current_lineno = current.instruction.node.lineno
    eof = get_last_line(cfg)
    # pg = program_graph.get_program_graph(code)
    # pdg_edges = extract_pdg_edges(pg)
    end_indicator = True # This flag here is to indicate if the it reaches the end
                            ## of the loop, if yes and the next node is the loop node
                            ## it should go back to the loop node. Manually set to False 
                            ## to end the loop if next node from function next_statement_predictor set it to false.
    pg = ""
    variable_table = {}
    while (end_indicator==True):
        # print(current_lineno)
        current_lineno = current.instruction.node.lineno
        curr_node = helper.find_node_by_line(tree, current_lineno)
        current_node_type = type(curr_node)
        execution_trace.append(current_lineno)
        if isinstance(current_node_type, (gg.For, gg.While, gg.If)):
            trace_till_last_branch = execution_trace
        else:
            trace_till_last_branch = []
        print(current_lineno)
        next = None
        ## changed current ---> curr_node now I am passing AST node into listing2
        if isinstance(current_node_type, gg.Expr) and \
           isinstance(current.instruction.node.value, gg.Call):
        #    print("we shouldn't be here")
           next = get_target_function(curr_node, cfg, call_edges, 'call')
        else:
            if current_node_type in COMPOUND_BRANCH_STMT_NODES:
                next = listing2.next_statement_predictor(cfg, pg, execution_trace, curr_node, current_lineno, program_code=code,program_trace=execution_trace, ast_tree = tree)
                if next is True:
                    print("this is current line")
                    print(current.instruction.node.lineno)
                    print(current.next)
                    # next = get_next_in_control_flow_order(current, cfg, call_edges)
                    # next = curr_node.next
                    next = list(current.next)[0]
                    # next = helper.find_cfg_node_by_line(cfg, next.lineno)
                    print(next.instruction.node.lineno)
                    # print("this is value of next")
                    # print(next)
                    # print(vars(cfg))
                    
                    # print(vars(next))

                    ### yesterday night stopped here:
                    ### add: ast node -> cfg
                    ### construct the program based on execution trace (remove the condition statement)
                    ### proper extraction from GPT answer
                    ### 
                else:
                    if "orelse" in vars(curr_node):
                        print("this 1")
                        next = curr_node.orelse
                        if next == []:
                            next = None
                        else:
                            # print(next)
                            next = next[0]
                            next = helper.find_cfg_node_by_line(cfg, next.lineno)
                    else: 
                        print("this 2")
                        next = get_next_in_control_flow_order(current, cfg, call_edges)
                        print("need to end the for loop")
                        
            else:
                print("this 3")
                if list(current.next) == []:
                    next = None
                else:
                    next = list(current.next)[0]

        if end_indicator == False:
            next = None
        current = next
        if(next is None):
            break
        print(current)
        current_lineno = current.instruction.node.lineno
    execution_trace.append(len(helper.split_lines_to_list(code)) +1)
    ### calling LLM for last body execution using variable table
    if variable_table:
        execution_trace = execution_trace[len(trace_till_last_branch):]
        last_brach_body_trace = run_GPT_for_last_execution_trace(code,execution_trace, variable_table)
        execution_trace = trace_till_last_branch + last_branch_body_trace
    return execution_trace

print(run_predictive_execution(TEST_PROGRAM1))
