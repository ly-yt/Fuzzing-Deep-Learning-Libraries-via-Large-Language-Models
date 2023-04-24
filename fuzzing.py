import json
import time
import openai
from typing import List
import torch
import tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm.codex import generate_seeds
from llm.example_batched_incoder_usage import InfillingModel

    
#掩盖填充的简陋版（不判断各个突变算子掩盖后生成的填充代码的正确性，直接对目标api应用每个突变算子进行代码突变）
import re

def Mask(seeds:List[str],target_api,op):
    # print(code)
    for code in seeds:
        slices = code.split('\n')
        for i in range(len(slices)):
            if op == 'method':
                if target_api in slices[i]:
                    s_list = slices[i].split(target_api)
                    slices[i] = "<insert>".join(s_list)
            elif op == 'argument':
                if target_api in slices[i]:
                    s_list = slices[i].split('(')
                    s_list[-1] = '<insert>)'
                    slices[i] = ''.join(s_list)
            elif op == 'prefix':
                if target_api not in slice[i] and 'tensorflow' not in slices[i]:
                    slices[i] = '<insert>'
                if target_api in slice[i]:
                    break
            else:
                if target_api in slices[i]:
                    slices[i+1] = '<insert>'
                    break
        code_mask = "\n".join(slices)
        dict_target_api[op].append(code_mask)

def generate(op,model):
    all_parts = [code_mask.split("<insert>") for code_mask in dict_target_api[op]]
    all_results = model.batched_infill(all_parts, max_to_generate=128, temperature=0.2)
    infill = [result['text'] for result in all_results]
    return infill

def Infill(seeds:List[str],target_api):
    MutationOp = ['argument','method','suffix','prefix']
    argument_infill,method_infill,suffix_infill,prefix_infill = [],[],[],[]
    infilling_model = InfillingModel("facebook/incoder-1B", cuda=True, half=False)
    for op in MutationOp:
        Mask(seeds=seeds,target_api=target_api,op=op)
        dict_target_api[op] = generate(op=op,model=infilling_model)


"""
Evolutionary fuzzing algorithm
简陋版——没有静态分析突变代码的正确性；
没有计算变异算子对于代码的先验和后验概率以此来选择最合适的变异算子；
没有对变异代码进行打分直接加入代码种子库中
"""
def EvoFuzz(API, Seeds:List[str], T_Budget):  #T_Budget默认为1分钟
    # 设定循环的时限T_Budget（1分钟）
    end_time = time.time() + 60 * T_Budget
    # 开始无限循环
    # initial_seeds = Seeds
    while True:
        # 在循环体内执行代码突变
        print("循环中...")
        Infill(Seeds,API)
        Seeds = Seeds + dict_target_api['argument'] + dict_target_api['method'] + dict_target_api['prefix'] + dict_target_api['suffix']
        # 判断是否超过了时限
        if time.time() > end_time:
            break
    return Seeds
    # 循环结束
    print("循环结束")
if __name__ == "__main__":
    target_api = 'random.normal'
# op = 'method'
    dict_target_api = {'method':[],'argument':[],'prefix':[],'suffix':[]}
    code = ["import tensorflow as tf\ninput_data = tf.random.normal([1, 2, 2, 1])\nfilters = tf.random.normal([2, 2, 1, 1])\nstrides = [1, 1, 1, 1]\npadding = 'SAME'\noutput = tf.nn.conv2d(input_data, filters, strides, padding, data_format='NHWC',dilations=None, name=None)\nreturn output"]
    Seeds_bank = EvoFuzz(target_api,code,10) #目标api最终经过突变后生成的测试代码库
