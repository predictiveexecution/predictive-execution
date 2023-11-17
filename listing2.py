import ast
from collections import Counter
from operator import itemgetter
import python_graphs
from python_graphs import program_graph
from python_graphs import program_graph_dataclasses as pb
import run.py

def extract_variables_from_expression(expression):
    variables = set()
    for node in ast.walk(expression):
        if isinstance(node, ast.Name):
            variables.add(node.id)
    return variables

def find_condition_expression(node):
    # Check if the node is an If, For, or While statement
    if isinstance(node, (ast.If, ast.For, ast.While)):
        return node.test

    # If the node is not a control statement, recursively search its children
    for child_node in ast.iter_child_nodes(node):
        condition_expression = find_condition_expression(child_node)
        if condition_expression:
            return condition_expression
    return None  # Condition expression not found    

def create_human_readable_condition(condition_expression):
    # Map operator names to more readable symbols
    operator_mapping = {
        'Eq': '==',
        'NotEq': '!=',
        'Lt': '<',
        'LtE': '<=',
        'Gt': '>',
        'GtE': '>=',
        'Is': 'is',
        'IsNot': 'is not',
        'In': 'in',
        'NotIn': 'not in',
    }
    if isinstance(condition_expression, ast.BoolOp):
        op = "and" if isinstance(condition_expression.op, ast.And) else "or"
        left_str = create_human_readable_condition(condition_expression.values[0])
        right_str = create_human_readable_condition(condition_expression.values[1])
        return f"{left_str} {op} {right_str}"
    elif isinstance(condition_expression, ast.Compare):
        left_str = get_value_or_id(condition_expression.left)
        op = operator_mapping.get(condition_expression.ops[0].__class__.__name__, '')
        right_str = get_value_or_id(condition_expression.comparators[0])
        return f"({left_str} {op} {right_str})"
    else:
        return ast.dump(condition_expression)
def get_value_or_id(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        return node.id
    else:
        return ast.dump(node)

def extract_variable_flow_from_pdg(pdg, code):
    graph = pdg
    variable_flow = {}

    for edge in graph.edges:
        if edge.type in [pb.EdgeType.LAST_READ, pb.EdgeType.LAST_WRITE]:
            from_line = graph.get_node(edge.id1).ast_node.lineno - 1
            var_from = graph.get_node(edge.id1).ast_node.id
            to_line = graph.get_node(edge.id2).ast_node.lineno - 1
            var_to = graph.get_node(edge.id2).ast_node.id
            assert var_from == var_to

            if from_line == to_line:
                continue
            if from_line in variable_flow:
                temp = variable_flow[from_line]
                variable_flow[from_line] = temp + [(var_from, to_line)]
            else:
                variable_flow[from_line] = [(var_from, to_line)]

    for cur_statement, from_statements in variable_flow.items():
        variable_flow[cur_statement] = sorted(list(from_statements), key=itemgetter(1))

    return variable_flow

def build_trace_with_occurrences(trace):
    trace_with_occurrences = []
    for item in trace:
        occurrence = 1
        while [item, occurrence] in trace_with_occurrences:
            occurrence += 1
        trace_with_occurrences.append([item, occurrence])
    return trace_with_occurrences

def extract_dynamic_slice(slicing_criterion, program_trace, variable_flow):
    fwd_slice, bckwd_slice = [], []
    temp = 0
    # print("Slicing criterion: ", slicing_criterion)
    trace_with_occurrences = build_trace_with_occurrences(program_trace)
    # print("Trace with occurrences", trace_with_occurrences)
    # print("Variable flow: ", variable_flow)
    # Find the last occurrence of the slicing criterion in the trace
    last_occurrence = None
    last_statement = program_trace[len(program_trace)-1]
    for item in reversed(variable_flow):
        if item == last_statement:
            #print(variable_flow[item])
            for sub_item in variable_flow[item]:
                if sub_item[0] == slicing_criterion:
                    print("occurrence of a: ",sub_item[1])
                    if(sub_item[1]!=temp and sub_item[1] in program_trace):
                        fwd_slice.append(sub_item[1])
                        temp = sub_item[1]
                    else:
                        temp = sub_item[1]
                        continue
                else:
                    continue            
        #print("Forward Slice",fwd_slice)
        return fwd_slice[::-1]

def get_sliced_code(program_code, backward_slice):
    lines = program_code.split('\n')
    sliced_code = [lines[line] for line in backward_slice[::-1]]
    return '\n'.join(sliced_code)

def extract_variables_values(variables, trace):
    variable_values = []
    reversed_trace = trace[::-1]  # Reverse the trace
    for variable in variables:
        last_value = None
        for line in reversed_trace:  # Iterate over the reversed trace
            if f'{variable} :' in line:
                value_start = line.find(f'{variable} :') + len(f'{variable} :')
                value_end = line.find('</state>', value_start)
                last_value = line[value_start:value_end].strip()
                # Remove additional information, if present
                last_value = last_value.split('<dictsep>')[0].strip()
                # Break once the last value is found
                break
        # If a value was found, add it to the result list
        if last_value is not None:
            variable_values.append((variable, last_value))
    return variable_values

def evaluate_condition(condition, variable_values):
    # Remove 'if' part of the statement, if present
    condition = condition.replace('if', '')
    #print("condition: ", condition)
    try:
        return eval(condition, dict(variable_values))
    except Exception as e:
        # print(f"Error evaluating condition: {e}")
        return False


def next_statement_predictor(cfg, pdg, exTrace, statement, line_number):
    condition_expression = find_condition_expression(statement)
    if condition_expression:
        # Extract variables from the condition expression
        variables_in_condition = extract_variables_from_expression(condition_expression)
        # print("Variables in condition:", variables_in_condition)
    else:
        # print("No condition expression found in the provided AST node.")
    variable_and_values = []
    if variables_in_condition not None:
        for variable in variables_in_condition:
            slicing_criterion = variable  # Change to 'b' if needed
            variable_flow = extract_variable_flow_from_pdg(pdg, program_code)
            slice_result = extract_dynamic_slice(slicing_criterion, program_trace, variable_flow)
            #print(f"Predictive backward slice with slicing criterion '{slicing_criterion}': {slice_result}")
            sliced_code = get_sliced_code(program_code, slice_result)
            # print("Sliced Code:")
            # print(sliced_code)
            '''change the js file contents in dataset.py and call code executor'''
            trace = run.main['trace']
            variable_value = extract_variables_values(variable, trace)
        variable_values.append[variable_value]
        if evaluate_condition(condition, variable_values):
            # print("The condition is True!")
            return line_number + 1
        else:
            # print("The condtion is False!")
            return line_number + 2 #codenetmut dataset
            