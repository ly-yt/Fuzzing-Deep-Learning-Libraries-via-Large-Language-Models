import os
import json
import openai

"""
背景：这些model的api接口都是需要付费的，每个人有18$的免费使用限额。我的free usage到2023.6.1到期。
"""
"""
下述model中text-davinci-003 is the Most capable GPT-3 model. Can do any task the other models can do.
其最长接受4,000 tokens输入
"""
"""
下述模型中gpt-3.5-turbo是chatgpt模型的api接口，是GPT-3.5系列中最快速、最便宜、最灵活的模型。
"""

openai.api_key = "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" #openai的api_key

#下述请求text-davinci-003的回应
response = openai.Completion.create(
  model="text-davinci-003",
#   engine = "code-davinci-003",
  prompt="\"\"\"\n0、create a function named f to do the following steps 1、Import TensorFlow 2.10.0 2、Generate input data 3、Call the API tf.nn.conv2d(input,filters,strides, padding,data_format='NHWC',dilations=None,name=None)\n\"\"\"",
  temperature=0,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

#下述请求chatgpt的回应
response1 = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
#   engine = "code-davinci-003",
messages=[{"role": "user", "content": "\"\"\"\n1、Import TensorFlow 2.10.0 2、Generate input data 3、Call the API tf.nn.conv2d(input,filters,strides, padding,data_format='NHWC',dilations=None,name=None)\n\"\"\""}],
temperature=0,
max_tokens=256,
top_p=1,
frequency_penalty=0,
presence_penalty=0
)
# print(response1)

#下述是gpt3系列模型获取代码结果的方式
# result_code = json.loads(str(response))['choices'][0]['text']  #str
# # print(response['choices'])
# print(type(result_code))

#下述是基于gpt3.5系列的chatgpt模型获取代码结果的方式
result_code = json.loads(str(response1))['choices'][0]['message']['content']  #str
# print(response['choices'])
print(result_code)

def write_code(result_code):
    #将生成的code写入code_generate.txt中
    f = open (r"/Users/lyt/vscode/llm/code_generate.txt","a",encoding="UTF-8")
    #写入
    f.write(result_code)
    #关闭
    f.close()
# write_code(result_code)