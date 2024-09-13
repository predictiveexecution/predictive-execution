import ast
import gast.gast as gg
from collections import Counter
from operator import itemgetter
import python_graphs
from python_graphs import program_graph
from python_graphs import program_graph_dataclasses as pb
import run
import helper
from prompt import *
def extract_variables_from_expression(node, expression):
    if isinstance(node,  (gg.While, gg.If)):
        variables = set()
        # print(vars(node))
    
        for i in ast.walk(expression):
            # print(vars(i))
            # print(i)
            if isinstance(i, gg.Name):
                variables.add(i.id)
        return variables
    elif isinstance(node,gg.For):
        variables = set()
        for i in ast.walk(node.target):
            if isinstance(i,gg.Name) and not isinstance(i.value,  gg.Call):
                variables.add(i.id)
        for i in ast.walk(node.iter):
            if isinstance(i,gg.Name):
                variables.add(i.id) 
        return variables

def find_condition_expression(node):
    # in this function, we 
    if isinstance(node, (gg.If,gg.While)):
        return node.test
    if isinstance(node, gg.For):
        return node

    # If the node is not a control statement, recursively search its children
    ### PROBLEM:
    #### Current PDG does not build corresponding call edge for gg.Match statement, needs to 
    #### put temporialy hold on this part of the code:
    # for child_node in ast.iter_child_nodes(node):
    #     condition_expression = find_condition_expression(child_node)
    #     if condition_expression:
    #         return condition_expression
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
    if isinstance(condition_expression, gg.BoolOp):
        op = "and" if isinstance(condition_expression.op, gg.And) else "or"
        left_str = create_human_readable_condition(condition_expression.values[0])
        right_str = create_human_readable_condition(condition_expression.values[1])
        return f"{left_str} {op} {right_str}"
    elif isinstance(condition_expression, gg.Compare):
        left_str = get_value_or_id(condition_expression.left)
        op = operator_mapping.get(condition_expression.ops[0].__class__.__name__, '')
        right_str = get_value_or_id(condition_expression.comparators[0])
        return f"({left_str} {op} {right_str})"
    else:
        return gg.dump(condition_expression)
def get_value_or_id(node):
    if isinstance(node, gg.Constant):
        return node.value
    elif isinstance(node, gg.Name):
        return node.id
    else:
        return gg.dump(node)

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
## can change to LLM needs validation.
def evaluate_condition(condition, variable_values,statement):
    # Remove 'if' part of the statement, if present
    if isinstance(statement, gg.If):
        condition = condition.replace("elif","").replace('if', '').replace(":",'').strip()
        print("condition: ", condition)
        # eval = python evaluator
        try:
            return eval(condition,{}, dict(variable_values))
        except Exception as e:
            print(f"Error evaluating condition: {e}")
            return False
    if isinstance(statement, gg.While):
        condition = condition.replace('while', '').replace(":",'').strip()
        print("condition", condition)
        try:
            return eval(condition,{}, dict(variable_values))
        except Exception as e:
            print(f"Error evaluating condition: {e}")
            return False
def extract_condition(condition_expression):
    print(condition_expression)

def next_statement_predictor(cfg, pdg, exTrace, statement, line_number, program_code, program_trace, ast_tree):

    condition_expression = find_condition_expression(statement)

    # prompt_code = helper.split_lines_to_list(program_code)[:statement.lineno-1]
    # print(program_code)
    if condition_expression:
        # Extract variables from the condition expression
        variables_in_condition = extract_variables_from_expression(statement,condition_expression)
        # print(variables_in_condition)

    variable_values = {}
    condition = ''
    program_code = helper.split_lines_to_list(program_code)
    print("this is program trace")
    print(program_trace)
    if variables_in_condition is not None:
        for variable in variables_in_condition:
            slicing_criterion = variable  # Change to 'b' if needed
            # variable_flow = extract_variable_flow_from_pdg(pdg, program_code)
            # slice_result = extract_dynamic_slice(slicing_criterion, program_trace, variable_flow)
            '''change the js file contents in dataset.py and call code executor'''
            # trace = run.main['trace'] ## LLM/PLM
            prompt_code = []
            
            print(program_code)
            for i in program_trace:
                ### need to see if the type is ggfor ggif ggwhile, we throw them away
                ### if/while a group, for will be another group.
                temp_node = helper.find_node_by_line(ast_tree, i)
                if i == temp_node.lineno and isinstance(temp_node, (gg.If, gg.While)):
                    continue
                elif i == temp_node.lineno and isinstance(temp_node, gg.For):
                    prompt_code.append(program_code[i-1].replace("for", ""))
                    continue
                prompt_code.append(program_code[i-1])
            user_prompt = predex_initial_user_prompt.replace("<\code>", ''.join(prompt_code).replace("\t",'')).replace("<\variable>", ''.join(variable)) 
            print(user_prompt)
            # print(f"predicting execution of \n{''.join(prompt_code)}\n with variable {variable}")
            pred_val = helper.llm(sys_prompt="",usr_prompt=user_prompt, engine='gpt-4o')
            print(pred_val)
            if "ERROR" in pred_val:
                print("Error in prediction")
                return False
            # print(trace)
            pred_val = pred_val.strip("`")
            # print(pred_val)
            #### temp comment out next 3 lines
            # condition = extract_condition(condition_expression)
            # variable_value = extract_variables_values(variable, program_trace) 
            try:
                variable_values[variable] = int(pred_val)
            except:
                variable_values[variable] = pred_val
            ## ValEvaluator
            # print()
        # print(variable_values)
        # print(statement.lineno)
        # program_code = helper.split_lines_to_list(program_code)
        condition = program_code[statement.lineno-1]
        # print(condition)
        ####
        #### Missing: bring variable table back to the main function.
        if evaluate_condition(condition, variable_values, statement):
            print(evaluate_condition(condition, variable_values, statement))
            return True
        else:
            print(evaluate_condition(condition, variable_values, statement))
            return False
def run_GPT_for_last_execution_trace(program_code, program_trace, variable_table):
    program_code = helper.split_lines_to_list(program_code)
    prompt_code = []
    for i in program_trace:
        prompt_code.append(program_code[program_trace[i]-1])
    
    user_prompt = predex_final_user_prompt.replace("<\code>", ''.join(prompt_code).strip("\t")).replace("<\variable>", str(variable_table))
    while(type(last_trace)!=list):
        
        last_trace = helper.llm(sys_prompt="",usr_prompt=user_prompt, engine='gpt-4')
    return last_trace
        
