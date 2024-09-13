import json
import re
import os
import time
import json
import logging
import hashlib
import html
import tempfile
import argparse
import subprocess
from pathlib import Path
from difflib import SequenceMatcher
from tqdm import tqdm
import openai
import gast
import networkx as nx
import python_graphs
import numpy as np
def run_compiler(code):
    if code.strip() == "":
        return ""
    tmp_dir = tempfile.TemporaryDirectory()
    md5_v = hashlib.md5(code.encode()).hexdigest()
    short_filename = "func_" + md5_v + ".java"
    with open(tmp_dir.name + "/" + short_filename, 'w') as f:
        f.write(code)

    try:
        compiler_run = subprocess.run(f'javac -Werror {tmp_dir.name}/{short_filename}',shell=True, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        tmp_dir.cleanup()
        return e.stderr.replace(f"{tmp_dir.name}/{short_filename}:","").replace(short_filename.replace('.java',''),'')
    tmp_dir.cleanup()
    return compiler_run.stdout

def llm(usr_prompt,stop = None, sys_prompt='',  seed= 1, temperature = 0.7, n = 1, max_token = None, engine = 'gpt-35'):
    import os
    from openai import AzureOpenAI
    client = AzureOpenAI(
    # api_key=os.environ.get("your api key"),
    api_key ='',
    api_version="",
    azure_endpoint=""
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
                "role": "user",
                "content": usr_prompt
            }
        ],
        stop = stop,
        model=engine,
    )
    # token_track(chat_completion)
    return chat_completion.choices[0].message.content

def split_lines_to_list(graph):
    with open('temp.txt', 'w') as infile:
        infile.write(str(graph))
    with open("temp.txt", 'r') as read_file:
        index = read_file.readlines()
    # subprocess.run(f"rm -rf /temp.txt", shell=True)
    return index

def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return "" # so we will return empty to remove the comment
        else: # otherwise, we will return the 1st group
            return match.group(1) # captured quoted-string
    return regex.sub(_replacer, string)

# def extract_variable_values_from_chat_gpt(string):

def find_cfg_node_by_line(cfg, target_lineno):
    """
    Traverse the entire CFG, call back node when target CFG node.lineno matches with target_lineno.
        Parameters:
        -- cfg: entire CFG object python-graphs
        -- target_lineno: target line number to search for in the CFG.
    """
    visited = set()
    stack = [cfg.get_start_control_flow_node()]

    while stack:
        node = stack.pop()
        if node in visited:
            continue

        visited.add(node)

        if node.instruction.node.lineno == target_lineno:
            return node

        for neighbor in node.next:
            if neighbor not in visited:
                stack.append(neighbor)

    return None  # Return None if no node with the target line number is found
    


           
            

def find_node_by_line(ast_tree, target_line):
    """
    Traverse the entire GAST tree and find the node corresponding to the target line number.

    Parameters:
    - ast_tree: The root of the GAST tree.
    - target_line: The line number to search for in the AST.

    Returns:
    - The first node that matches the target line number, or None if no match is found.
    """
    def traverse_ast(node):
        # Check if the node has a line number and if it matches the target line number
        if hasattr(node, 'lineno') and node.lineno == target_line:
            return node
        
        # Recursively traverse the child nodes
        for child in gast.iter_child_nodes(node):
            result = traverse_ast(child)
            if result is not None:
                return result
        
        # Return None if no matching node is found
        return None
    
    # Start traversal from the root of the AST
    return traverse_ast(ast_tree)

def code_to_ast(code):
    """
    Convert a Python code snippet to an Abstract Syntax Tree (AST) using the gast library.

    Parameters:
    - code: The Python code snippet to parse.

    Returns:
    - The root node of the AST.
    """
    return gast.parse(code)
def traverse_ast(tree):
    """
    Traverse the entire GAST tree and yield each node in depth-first order.

    Parameters:
    - tree: The root of the GAST tree.

    Yields:
    - Each node in the GAST tree in depth-first order.

    Returns:
    - All nodes in the GAST tree in depth-first order in a list.
    """
    def traverse(node):
        yield node
        for child in gast.iter_child_nodes(node):
            yield from traverse(child)
    
    return list(traverse(tree))







                    
