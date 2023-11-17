# Predictive Execution: 
In this paper, we advocate for an execution paradigm called predictive execution. In predictive execution, with a specific input, the execution is not carried out with the computer performing the instruction in the program. Instead, a trained machine learning model predicts the execution steps and as a result, the execution trace corresponding to the input is derived without actual execution.
The predictive execution paradigm has the potential to improve efficiency and effectiveness in software testing, fault localization, fuzzing testing, and early bug detection. We propose PredEx, a predictive executor for Python programs. The principle of PredEx is
a blended analysis between program analysis and Large Language Model (LLM). We break down the task of predicting the execution into smaller sub-tasks and leverage the deterministic nature when an execution order can be deterministically decided. When it is not certain, we use predictive backward slicing and the predicted
values to help an LLM pay attention to the statements affecting the values at the conditions to decide on the next statement. Our empirical evaluation on real-world programs shows that PredEx achieves relatively higher accuracy up to 26% in predicting full execution traces than the state-of-the-art model. PredEx
also produces 6.8%–26.5% more precise execution traces than the baseline model. In predicting the next executed statement, its relative improvement over the baseline is 13.5%–15.8%. We also show the usefulness of PredEx in two applications: static code coverage computation and static detection of run-time exceptions.

## Dataset Links
Here are the links for datasets used in this paper:
  1. CodeNetMut dataset (Chenxiao Liu, Shuai Lu, Weizhu Chen, Daxin Jiang, Alexey Svyatkovskiy, Shengyu Fu, Neel Sundaresan, and Nan Duan. 2023. Code Execution with Pre-trained Language Models. arXiv:2305.05383 [cs.PL])
  2. BugsInPy dataset (Ratnadira Widyasari, Sheng Qin Sim, Camellia Lok, Haodi Qi, Jack Phan, Qijin Tay, Constance Tan, Fiona Wee, Jodie Ethelda Tan, Yuheng Yieh, et al. 2020. Bugsinpy: a database of existing bugs in python programs to enable controlled testing and debugging studies. In Proceedings of the 28th ACM joint meeting on
european software engineering conference and symposium on the foundations of software engineering. 1556–1560.)
  3. Java dataset for method-level vulnerability detection: [link](https://drive.google.com/drive/folders/1LVlQJz4sXkByJS9FUW61ecSv_jWI75E_?usp=sharing)

## Getting Started with PredEx
### Run Instructions
  
```
Input your code in listing 1.py and run the file. 
```
