import json
import os
import re
import time
import openai
from typing import List
import torch
import tokenizers
import numpy as np
# import matplotlib.pyplot as plt
from collections import Counter
from pyflowchart import Flowchart
from numpy.random import beta
from transformers import AutoModelForCausalLM, AutoTokenizer
from codex import generate_seeds
from example_batched_incoder_usage import InfillingModel
# from llm.codex import generate_seeds
# from llm.example_batched_incoder_usage import InfillingModel

    

import re
#按批填充生成
# def batch_generate(op,model):
#     all_parts = [code_mask.split("<insert>") for code_mask in dict_target_api[op]]
#     all_results = model.batched_infill(all_parts, max_to_generate=128, temperature=0.2)
#     infill = [result['text'] for result in all_results]
#     return infill


#掩盖填充的简陋版（不判断各个突变算子掩盖后生成的填充代码的正确性，直接对目标api应用每个突变算子进行代码突变）
"""
Evolutionary fuzzing algorithm
简陋版——没有静态分析突变代码的正确性；
没有计算变异算子对于代码的先验和后验概率以此来选择最合适的变异算子；
没有对变异代码进行打分直接加入代码种子库中
"""
# def EvoFuzz(API, Seeds:List[str], T_Budget):  #T_Budget默认为1分钟
#     # 设定循环的时限T_Budget（1分钟）
#     end_time = time.time() + 60 * T_Budget
#     # 开始无限循环
#     # initial_seeds = Seeds
#     while True:
#         # 在循环体内执行代码突变
#         print("循环中...")
#         Infill1(Seeds,API)
#         Seeds = Seeds + dict_target_api['argument'] + dict_target_api['method'] + dict_target_api['prefix'] + dict_target_api['suffix']
#         # 判断是否超过了时限
#         if time.time() > end_time:
#             break
#     return Seeds
#     # 循环结束
#     print("循环结束")

class Fuzzing:
    def __init__(self,target_api=None,code=None):
        self.dict_code = {} #代码突变后对每个突变代码进行打分
        self.infilling_model = InfillingModel("facebook/incoder-1B", cuda=True, half=False) #incoder模型
        self.dict_target_api = {'method':[],'argument':[],'prefix':[],'suffix':[]} #各个突变算子对应的突变代码
        self.target_api = target_api #目标api
        self.code = code #初始种子代码

    #应用突变算子进行代码掩盖  
    def Mask(self,seeds:List[str],target_apis:List[str],op):
    # print(code)
        for code in seeds:
            slices = code.split('\n')
            for i in range(len(slices)):
                if op == 'method':
                    for target_api in target_apis:
                        if target_api in slices[i]:
                            s_list = slices[i].split(target_api)
                            slices[i] = "<insert>".join(s_list)
                            break
                elif op == 'argument':
                    for target_api in target_apis:
                        if target_api in slices[i]:
                            s_list = slices[i].split('(')
                            s_list[-1] = '<insert>)'
                            slices[i] = ''.join(s_list)
                            break
                elif op == "prefix":
                    for target_api in target_apis:
                        if target_api in slices[i]:
                            slices[i-1] = '<insert>'
                            break
                else:
                    #此步暂不考虑目标函数在末尾的情况
                    for target_api in target_apis:
                        if target_api in slices[i]:
                            if i < len(slices)-1:
                                slices[i+1] = '<insert>'
                                break          
            code_mask = "\n".join(slices)
            self.dict_target_api[op].append(code_mask)

    #代码填充
    def generate1(self,op,model):
        all_parts = [code_mask.split("<insert>") for code_mask in self.dict_target_api[op]]
        all_results = [model.infill(parts,max_to_generate=128, temperature=0.2) for parts in all_parts]
        # all_results = model.batched_infill(all_parts, max_to_generate=128, temperature=0.2)
        infill_result = [result['text'] for result in all_results]
        # print('infill_result:',infill_result)
        return infill_result

    #此段代码要改进，如何继续变异变异后的代码(已改进)
    #对种子代码应用每个突变算子进行代码突变
    def Infill1(self,seeds:List[str],target_api:List[str]):
        MutationOp = ['argument','method','suffix','prefix']
        for op in MutationOp:
            self.Mask(seeds=seeds,target_apis=target_api,op=op)
            self.dict_target_api[op] = self.generate1(op=op,model=self.infilling_model)

    #初始化每个突变算子的先验概率
    def InitializeMPrior(self):
        dic_op = {'argument':{'success':1,'fail':1},'method':{'success':1,'fail':1},'prefix':{'success':1,'fail':1},'suffix':{'success':1,'fail':1}}
        return dic_op

    #将生成的突变代码写入文件
    def write_code(self,code):
        #直接写入字符串数据
        with open('pylint_code.py','w') as f:
            f.writelines(code)

    #计算每个突变算子的beta分布
    def beta_(self,a,b):
        xs = beta(a, b)
        return xs
    
    #更新每个突变算子的后验概率
    def UpdateMPosterior(self,op,dic_op):
        code_list = list(set(self.dict_target_api[op])) #去重后的代码列表
        for code in code_list:
            self.write_code(code)
            os.system("pylint --errors-only /Users/lyt/vscode/llm/pylint_code.py > /Users/lyt/vscode/llm/test.txt")#通过命令行方式执行pylint命令
            file_size = os.path.getsize('./llm/test.txt')
            if not file_size:
                dic_op[op]['success'] +=1
            else:
                dic_op[op]['fail'] +=1
                self.dict_target_api[op].remove(code)  #删除静态分析后错误的代码
        xs = self.beta_(dic_op[op]['success'],dic_op[op]['fail'])
        return xs

    #选择当前种子代码应用各个突变算子突变后最佳突变代码所对应的突变算子
    def SelectMutationOp(self):
        MutationOp = ['argument','method','suffix','prefix']
        dic_op = self.InitializeMPrior()
        p = [self.UpdateMPosterior(op,dic_op) for op in MutationOp]
        index = p.index(max(p)) 
        return MutationOp[index]  #选择出对于该目标api使代码变异最优解的突变算子

    #计算代码数据流图深度
    def flowchart_code(self,code):
        fc = Flowchart.from_code(code)
        result = fc.flowchart()
        depth = len(result.split('->')) - 1
        return depth

    #统计突变代码中包含不同api的个数
    def different_api(self,code):
        code_list = code.split('\n')
        api = [re.split(r"[.(]",code)[code.count('.')] for code in code_list if code.count('.')]
        c = Counter(api)
        count_differ_apis = len(dict(c).keys())
        count_same_apis = sum(list(dict(c).values()))-count_differ_apis
        return count_differ_apis - count_same_apis

    #根据代码数据流图深度和包含的api个数对突变代码进行打分
    def FitnessScore(self,valid_codes):
        for code in valid_codes:
            score = self.flowchart_code(code) + self.different_api(code)
            if score not in self.dict_code.keys():
                self.dict_code[score] = [code]
            else:
                if code not in self.dict_code[score]:
                    self.dict_code[score].append(code)

    #根据当前种子库中代码对应的分数分布，计算选择每个种子代码的概率 
    def softmax(self,x):
        # 计算每行的最大值
        row_max = np.max(x)
        # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
        x = x - row_max
        # 计算e的指数次幂
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp)
        probability = x_exp / x_sum
        return probability

    #选择概率最大的代码进行此轮代码突变
    def SelectSeeds(self):
        # seeds = list(dict_code.values())
        scores = list(self.dict_code.keys())
        pr = self.softmax(scores)
        index = np.argmax(pr)
        seed_selected = self.dict_code[scores[index]]
        return seed_selected

    #选择突变后的代码中包含的api来作为下一个目标api
    def determine_api(self,seeds_selected):
        apis = []
        remove_tokens = ['<|','from','import','class','#','assert','if']
        for seed in seeds_selected:
            code_list = seed.split('\n')
            for code in code_list:
                flag = True
                for token in remove_tokens:
                    if token in code:
                        flag = False
                        break
                if flag:
                    if code.count('.'):
                        api = re.split(r"[.(]",code)[code.count('.')]
                        apis.append(api)
        return list(set(apis))

    """
    Evolutionary fuzzing algorithm
    改进版——静态分析突变代码的正确性✅；
    计算变异算子对于代码的先验和后验概率以此来选择最合适的变异算子✅；
    对变异代码进行打分来选择所生成的适合的突变代码加入代码种子库中✅；
    之后对突变代码进行进一步变异✅
    """
    def EvoFuzz1(self,API:List[str], Seeds:List[str], T_Budget):  #T_Budget默认为1分钟
        # 设定循环的时限T_Budget（1分钟）
        end_time = time.time() + 60 * T_Budget
        # 开始无限循环
        # initial_seeds = Seeds
        initial = True
        count =0
        while True:
            count += 1
            # 在循环体内执行代码突变
            print("循环中...") 
            #以下插入根据代码打分函数得出的分数来选择代码种子的段落
            #todo
            if initial: #初次执行
                seeds_selected = Seeds
                initial = False
            else:
                # FitnessScore(dict_target_api[op_selected])
                seeds_selected = self.SelectSeeds()
                #自动对变异后的代码进行自动变异
                API = self.determine_api(seeds_selected)
                print('当前api是',API)
            #--------------------------
            self.Infill1(seeds_selected,API)
            # FitnessScore(dict_target_api['method'])
            op_selected = self.SelectMutationOp()
            #此处要做个判断，突变后的代码若不正确则继续上述操作重新突变
            if self.dict_target_api[op_selected]:
                # print('op_selected:',op_selected)
                # print('len(dict_target_api[op_selected]):',len(dict_target_api[op_selected]))
                self.FitnessScore(self.dict_target_api[op_selected])
                # print('dict_code_keys:',dict_code.keys())
                # seeds_selected = SelectSeeds()
                Seeds += self.dict_target_api[op_selected]
                Seeds = list(set(Seeds))  #对种子代码库进行去重
                if count >=3:
                    break
                # 判断是否超过了时限
                if time.time() > end_time:
                    break
            else:
                if count >=3:
                    break
                initial = True
                print('种子代码突变后的代码错误，重新突变')
            self.dict_target_api = {'method':[],'argument':[],'prefix':[],'suffix':[]}
        return Seeds
        # 循环结束
        




if __name__ == "__main__":
    target_api = ['random.normal']
    prompt = "\"\"\"\n 1、Import TensorFlow 2.10.0 2、Generate input data 3、Call the API tf.{}\n\"\"\"".format(target_api[0])
    #使用openai的codex模型根据上述提示生成种子代码
    code = [generate_seeds(prompt)]
    # op = 'method'
    # code = ["import tensorflow as tf\ninput_data = tf.random.normal([1, 2, 2, 1])\nfilters = tf.random.normal([2, 2, 1, 1])\nstrides = [1, 1, 1, 1]\npadding = 'SAME'\noutput = tf.nn.conv2d(input_data, filters, strides, padding, data_format='NHWC',dilations=None, name=None)\nreturn output"]
    fuzzing = Fuzzing(target_api,code)
    seeds_bank = fuzzing.EvoFuzz1(target_api,code,2)
    print('目标api：{}最终突变后的代码库为{}'.format(target_api[0],seeds_bank))
