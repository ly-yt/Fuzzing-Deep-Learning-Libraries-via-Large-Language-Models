from typing import List

import torch
import tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
tokenizers_version = tuple(int(n) for n in tokenizers.__version__.split('.'))
if tokenizers_version < (0, 12, 1):
    print("warning: Your tokenizers version looks old and you will likely have formatting issues. We recommend installing tokenizers >= 0.12.1")

# set BIG_MODEL to use the 6.7B parameter model
BIG_MODEL = False

# use a GPU
# CUDA = True
CUDA = False


# print intermediate outputs of infilling
VERBOSE = False

if BIG_MODEL:
    model_name = "facebook/incoder-6B"

    # the arguments added below will load a half precision version of the model,
    # which requires less RAM than loading the full float32 version.  this 
    # should fit in ~16GB of RAM
    # NOTE: half precision should *not* be used if you plan to fine-tune the
    # model. You'll need full precision and a lot of GPU memory. We have not
    # tested fine-tuning in `transformers` (the model was trained in fairseq)
    if CUDA:
        kwargs = dict(
            revision="float16", 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        kwargs = dict(
            low_cpu_mem_usage=True,
        )
else:
    model_name = "facebook/incoder-1B"
    kwargs = {}

print("loading model")
"""
下述模型加载是直接通过模型名作为下载接口，模型文件默认下载在/Users/lyt/.cache下
‼️下述很重要
问题：直接通过模型名作为接口调用下载的方式经常会报错，原因在于huggingface库在国外，
国内若无法连接外网或者下载速度过慢会导致模型未下载全就后续使用了。
解决方式：人工在huggingface官网下手动下载相应文件保持文件名不变，放入一个文件夹下，
调用的时候传入该文件夹对应路径。
(具体必要的下载文件可参考
https://getliner.com/zh/picked-by-liner/reader-mode?url=https%3A%2F%2Fzhuanlan.zhihu.com%2Fp%2F147144376)
"""
# model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
model = AutoModelForCausalLM.from_pretrained("/Users/lyt/vscode/llm/incoder_1B")
print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("/Users/lyt/vscode/llm/incoder_1B")
print("loading complete")

if CUDA:
    # if you plan to fine-tune the model, you should not use half precision.
    model = model.half().cuda()

# signals the start of a document
BOS = "<|endoftext|>"
# signals the end of a generated infill
EOM = "<|endofmask|>"

def make_sentinel(i):
    # signals (1) a location to insert an infill and (2) the start of the infill generation
    return f"<|mask:{i}|>"

def generate(input: str, max_to_generate: int=128, temperature: float=0.2):
    """
    Do standard left-to-right completion of the prefix `input` by sampling from the model
    通过从模型中采样，对前缀“input”执行标准的从左到右完成
    """
    input_ids = tokenizer(input, return_tensors="pt").input_ids
    if CUDA:
        input_ids = input_ids.cuda()
    max_length = max_to_generate + input_ids.flatten().size(0)
    if max_length > 2048:
        print("warning: max_length {} is greater than the context window {}".format(max_length, 2048))
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, do_sample=True, top_p=0.95, temperature=temperature, max_length=max_length)
    # pass clean_up_tokenization_spaces=False to avoid removing spaces before punctuation, e.g. "from ." -> "from."
    detok_hypo_str = tokenizer.decode(output.flatten(), clean_up_tokenization_spaces=False)
    if detok_hypo_str.startswith(BOS):
        detok_hypo_str = detok_hypo_str[len(BOS):]
    return detok_hypo_str

def infill(parts: List[str], max_to_generate: int=128, temperature: float=0.2, extra_sentinel: bool=True, max_retries: int=1):
    """
    Generate infills to complete a partial document, e.g.
    [A C E] -> [A B C D E], where B and D are infills that have been generated.
    parts: List[str]. list of parts of the document. One string will be
            inserted in between each element, i.e. infilling N-1 locations for a list
            of length N.
    max_to_generate: int. maximum number of tokens to generate. Keep in mind
            that the model context size is 2048.
    temperature: float. temperature parameter for sampling.
    extra_sentinel: bool. we recommend setting this to True, as it makes it
            easier for the model to end generated infills. See the footnote in 
            section 2.2 of our paper for details.
    max_retries: int. if > 1, use rejection sampling to keep sampling infills until
            all infills sample a completion token.
            （如果 > 1，则使用拒绝采样保持采样填充，直到所有填充样品完成令牌）
    returns a dictionary containing the following:
        text:  str, the completed document (with infills inserted)
        parts:  List[str], length N. Same as passed to the method
        infills:  List[str], length N-1. The list of infills generated
        retries_attempted:  number of retries used (if max_retries > 1)
    """
    assert isinstance(parts, list)   #assert语句是一种插入调试断点到程序的一种便捷的方式。加入assert后的代码若运行报错，将抛出assertion error后面的所有就不会执行。isinstance函数是判别一个变量的类型是否是某个类型。
    retries_attempted = 0
    done = False

    while (not done) and (retries_attempted < max_retries):
        retries_attempted += 1

        if VERBOSE:
            print(f"retry {retries_attempted}")
        
        ## (1) build the prompt
        if len(parts) == 1:
            prompt = parts[0]
        else:
            prompt = ""
            # encode parts separated by sentinel
            for sentinel_ix, part in enumerate(parts):
                prompt += part
                if extra_sentinel or (sentinel_ix < len(parts) - 1):
                    prompt += make_sentinel(sentinel_ix)
        
        infills = []
        complete = []

        done = True

        ## (2) generate infills
        for sentinel_ix, part in enumerate(parts[:-1]):
            complete.append(part)
            prompt += make_sentinel(sentinel_ix)
            # TODO: this is inefficient as it requires re-encoding prefixes repeatedly
            completion = generate(prompt, max_to_generate, temperature)
            completion = completion[len(prompt):]
            if EOM not in completion:
                if VERBOSE:
                    print(f"warning: {EOM} not found")
                completion += EOM
                done = False
            completion = completion[:completion.index(EOM) + len(EOM)]
            infilled = completion[:-len(EOM)]
            infills.append(infilled)
            complete.append(infilled)
            prompt += completion
        complete.append(parts[-1])
        text = ''.join(complete)

    if VERBOSE:
        print("generated text:")
        print(prompt)
        print()
        print("parts:")
        print(parts)
        print()
        print("infills:")
        print(infills)
        print()
        print("restitched text:")  #填充完毕后的text
        print(text)
        print()
    
    return {
        'text': text, # str, the completed document (with infills inserted)
        'parts': parts, # List[str], length N. Same as passed to the method
        'infills': infills, # List[str], length N-1. The list of infills generated
        'retries_attempted': retries_attempted, # number of retries used (if max_retries > 1)
    } 

def code_to_docstring(max_to_generate=128, temperature=0.2):
    # this will sometimes generate extra functions! this can be avoided by truncating generation when e.g. a """ is produced
    example = '''\
def count_words(filename):
    """ <insert> """
    counts = Counter()
    with open(filename) as file:
        for line in file:
            words = line.split(' ')
            counts.update(words)
    return counts'''

    parts = example.split("<insert>")
    result = infill(parts, max_to_generate=max_to_generate, temperature=temperature)
    print("completed document:")
    print(result["text"])
    return result["text"]

def docstring_to_code(max_to_generate=128, temperature=0.2):
    example = '''\
def <insert>
    """ Count the number of occurrences of each word in the file. """
    <insert>
<|/ file |>'''
    parts = example.split("<insert>")
    result = infill(parts, max_to_generate=max_to_generate, temperature=temperature)
    print("completed document:")
    print(result["text"])
    return result["text"]

if __name__ == "__main__":
    print("code to docstring test:")
    code_to_docstring()
    print()
    print("docstring to code test:")
    docstring_to_code()