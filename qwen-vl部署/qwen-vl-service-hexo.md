# Qwen2.5-VL-模型部署遇到的问题和解决方法

---
### 一、Qwen2.5-VL-模型介绍
最近遇到了一个根据图片生成描述的项目，之前使用过Qwen-VL 1.0 ，当时测试效果感觉还可以，当时还在一些任务上做了微调，
后来它不更新了，我也转到其他感兴趣的方向，就没有再跟了。最近要做一个根据图片生成描述的项目，看了Qwen-VL和MiniCPM，没想到Qwen-VL更新到2.5了，
之前印象还不错，就是它了。本次使用的是7B的模型，感觉效果已经满足需要了。

---
### 一、下载模型 
目前modelscope下载已经非常快速了，所以建议直接从modelscope下载。另外模型一般比较大，我下载的7B 有16G大小，所以建议在服务器上直接用代码下载，
省得后面还要上传。
```commandline
pip install modelscope
```
然后执行下面的python代码下载
```python
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct',cache_dir='./model/')
```
修改cache_dir参数，指定模型要保存的目录

---
### 二、环境配置
建议使用conda 创建新环境，以免影响现有程序的正常环境
conda 安装请参考xxxxxx
#### 创建新python环境
```commandline
conda create -n new_env python=3.10.16
```
小tips 配置pip为国内源可以加快后面的速度
pip配置国内源
1. 执行命令,新建文件夹
```commandline
cd && mkdir .pip && touch .pip/pip.conf && cd .pip
```
2. 编辑pip.conf文件，粘贴下面的内容
```commandline
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = https://pypi.tuna.tsinghua.edu.cn
```
#### 配置Qwen2.5-VL的环境
这一步主要参考项目的主页，我发现github的主页和modelscope的主页不太一样，我这边按照项目的github的主页安装的。
项目主页：https://github.com/QwenLM/Qwen2.5-VL
1. 配置最新版本的transformer版本
```shell
pip install git+https://github.com/huggingface/transformers accelerate
```
这一部必须要这样安装transformer，否则会遇到下面的错误
```commandline
KeyError: 'qwen2_5_vl'
```
这也是建议创建新环境的原因，之前的项目代码用最新版本的transformer库可能有问题。
2. 安装qwen-vl工具包
```shell
pip install qwen-vl-utils[decord]
```
3. 检测是否安装成功
```shell
pip freeze | grep -e trans -e acce -e qwen
```
执行命令之后输出如下
```commandline
accelerate==1.6.0
qwen-vl-utils==0.0.10
transformers @ git+https://github.com/huggingface/transformers@99f9f1042f59f60de9a8f0538c1117e4eca38ef9
```
4. 运行Qwen-VL的测试用例
上面3步没有问题，我们可以测试下Qwen-VL是否正常运行
```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 改成下载的模型路径
modelPath = "/workspace/work/tmp/model/Qwen/Qwen2___5-VL-7B-Instruct/"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    modelPath, torch_dtype="auto", device_map="auto"
)

# default processor
processor = AutoProcessor.from_pretrained(modelPath)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "请详细描述这个图片"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```
我们的指令是让模型“请详细描述这个图片”  
['这张图片展示了一位女性和一只狗在海滩上互动的温馨场景。女性坐在沙滩上，穿着格子衬衫和牛仔短裤，脚上穿着凉鞋。她的头发披散着，脸上带着微笑，显得非常开心。她正与一只浅棕色的狗进行互动，狗戴着彩色的项圈，看起来像是拉布拉多犬。狗伸出前爪，似乎是在和女性握手或玩耍。\n\n背景是广阔的海洋和天空，海浪轻轻拍打着沙滩。阳光从画面右侧斜射过来，给整个场景增添了一种温暖而柔和的光线效果。沙滩上的沙']

---
### 使用VLLM部署Qwen2.5—VL
1. 安装vllm框架  
通过上面几个步骤，我们可以在本地运行模型，下面我们通过部署服务，让其他人可以远程使用模型。
我们vllm来部署Qwen-vl模型，vllm这个架构可以提高大模型服务的性能，支撑更多的并发，用过的都说好。
我一开始的方法是使用pip直接安装vllm 命令如下
```shell
pip install vllm
```
但是编译特别慢，最后还是出错了，后来发现可以直接下载whl文件来部署。
https://pypi.org/project/vllm/
![whl1](/resources/0410-qwen-vl-serveice/whl1.png)
![whl1](/resources/0410-qwen-vl-serveice/whl2.png)
下载到本地，然后上传到服务器之后，在服务器安装whl文件
```shell
pip install vllm-0.8.3-cp38-abi3-manylinux1_x86_64.whl
```
这次非常迅速，几分钟就安装完了。
测试vllm是否安装成功
```shell
(common) [root@17:34:09 work]$ pip freeze | grep vllm
vllm @ file:///workspace/work/vllm-0.8.3-cp38-abi3-manylinux1_x86_64.whl#sha256=cc6fcaa88828b07ec315afe883f050c99c1a2026accd4101fd850419c44b27bd
```
显示这样就代表安装成功了
2. 使用vllm框架部署Qwen-vl
```shell
CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server \
--served-model-name Qwen2-VL-7B-Instruct \
--model /workspace/work/tmp/model/Qwen/Qwen2___5-VL-7B-Instruct/ \ 
--port 8080 &
```
CUDA_VISIBLE_DEVICES环境变量指定使用的GPU卡号，port参数指定端口号，这样服务就搭建好了
3. 测试服务
使用下面的测试程序
```python
import requests

def send_chat_request(model, messages, url="http://localhost:8000/v1/chat/completions"):
    """
    发送聊天请求到指定的服务端点。

    参数:
        model (str): 模型名称，例如 "Qwen2-VL-7B-Instruct"。
        messages (list): 消息列表，包含系统和用户消息。
        url (str): 服务端点，默认为 "http://localhost:8000/v1/chat/completions"。

    返回:
        dict: 服务器返回的 JSON 响应。
    """
    # 定义请求头
    headers = {
        "Content-Type": "application/json"
    }

    # 定义请求体
    data = {
        "model": model,
        "messages": messages
    }

    try:
        # 发送 POST 请求
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # 检查 HTTP 错误

        # 返回解析后的 JSON 响应
        return response.json()
    except requests.exceptions.RequestException as e:
        # 捕获并处理请求异常
        print("An error occurred:", e)
        return None

# 示例调用
if __name__ == "__main__":
    # 定义模型和消息
    model_name = "Qwen2-VL-7B-Instruct"
    chat_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
            {"type": "text", "text": "请详细描述这个图片"}
        ]}
    ]

    # 调用函数并打印结果
    result = send_chat_request(model_name, chat_messages, url = 'http://127.0.0.1:8888/v1/chat/completions')
    if result:
        print("Response Body:", result)
```
上面测试成功执行之后，和我们刚才本地调用应该是一致的。

---
### 踩的坑
1. cuda版本问题  
一开始在这个机器上用了原来用过的镜像，直接安装vllm，结果报下面的错误
```commandline
Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 34, in <module>
        File "/tmp/pip-install-akqe0eao/xformers_96c3e9f6110b43c786c3807c65d2ed03/setup.py", line 691, in <module>
          setuptools.setup(
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/setuptools/__init__.py", line 87, in setup
          return distutils.core.setup(**attrs)
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/setuptools/_distutils/core.py", line 185, in setup
          return run_commands(dist)
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/setuptools/_distutils/core.py", line 201, in run_commands
          dist.run_commands()
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/setuptools/_distutils/dist.py", line 969, in run_commands
          self.run_command(cmd)
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/setuptools/dist.py", line 1208, in run_command
          super().run_command(command)
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
          cmd_obj.run()
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/setuptools/command/install.py", line 68, in run
          return orig.install.run(self)
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/setuptools/_distutils/command/install.py", line 698, in run
          self.run_command('build')
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/setuptools/_distutils/cmd.py", line 318, in run_command
          self.distribution.run_command(command)
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/setuptools/dist.py", line 1208, in run_command
          super().run_command(command)
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
          cmd_obj.run()
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/setuptools/_distutils/command/build.py", line 132, in run
          self.run_command(cmd_name)
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/setuptools/_distutils/cmd.py", line 318, in run_command
          self.distribution.run_command(command)
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/setuptools/dist.py", line 1208, in run_command
          super().run_command(command)
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
          cmd_obj.run()
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/setuptools/command/build_ext.py", line 84, in run
          _build_ext.run(self)
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/setuptools/_distutils/command/build_ext.py", line 346, in run
          self.build_extensions()
        File "/tmp/pip-install-akqe0eao/xformers_96c3e9f6110b43c786c3807c65d2ed03/setup.py", line 648, in build_extensions
          super().build_extensions()
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/torch/utils/cpp_extension.py", line 552, in build_extensions
          _check_cuda_version(compiler_name, compiler_version)
        File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/torch/utils/cpp_extension.py", line 447, in _check_cuda_version
          raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
      RuntimeError:
      The detected CUDA version (11.7) mismatches the version that was used to compile
      PyTorch (12.4). Please make sure to use the same CUDA versions.
```
首先确定容器中使用的cuda版本
```commandline
nvcc --version
```
输出
```commandline
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Nov_22_10:49:45_PST_2023
Cuda compilation tools, release 11.7, V11.7.64
```
检测pytorch的cuda版本
```python
import torch
print(torch.version.cuda)
```
输出
```commandline
import torch
print(torch.version.cuda)
```
可以使用下面的命令，升级容器的cuda版本，保持一致
```commandline
sudo apt-get update
sudo apt-get install cuda-12-4
```
2. python包冲突，cuda正确之后，用的原来的python环境，先重新安装了Transform库，结果报包冲突
```commandline
Traceback (most recent call last):
  File "/workspace/work/Qwen-VL-master/test2.py", line 1, in <module>
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
  File "<frozen importlib._bootstrap>", line 1055, in _handle_fromlist
  File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/transformers/utils/import_utils.py", line 1956, in __getattr__
    value = getattr(module, name)
  File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/transformers/utils/import_utils.py", line 1955, in __getattr__
    module = self._get_module(self._class_to_module[name])
  File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/transformers/utils/import_utils.py", line 1969, in _get_module
    raise RuntimeError(
RuntimeError: Failed to import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl because of the following error (look up to see its traceback):
If you use `@root_validator` with pre=False (the default) you MUST specify `skip_on_failure=True`. Note that `@root_validator` is deprecated and should be replaced with `@model_validator`.

For further information visit https://errors.pydantic.dev/2.11/u/root-validator-pre-skip
```
所以建议新建一个干净的conda环境

3. 安装vllm特别慢 停留在下面的阶段
```commandline
Building wheels for collected packages: xformers
  Building wheel for xformers (setup.py) ...
```
```commandline
Traceback (most recent call last):
  File "/root/anaconda3/envs/monkey/lib/python3.9/runpy.py", line 197, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/root/anaconda3/envs/monkey/lib/python3.9/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/vllm/entrypoints/openai/api_server.py", line 531, in <module>
    asyncio.run(run_server(args))
  File "/root/anaconda3/envs/monkey/lib/python3.9/asyncio/runners.py", line 44, in run
    return loop.run_until_complete(main)
  File "/root/anaconda3/envs/monkey/lib/python3.9/asyncio/base_events.py", line 647, in run_until_complete
    return future.result()
  File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/vllm/entrypoints/openai/api_server.py", line 498, in run_server
    async with build_async_engine_client(args) as async_engine_client:
  File "/root/anaconda3/envs/monkey/lib/python3.9/contextlib.py", line 181, in __aenter__
    return await self.gen.__anext__()
  File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/vllm/entrypoints/openai/api_server.py", line 110, in build_async_engine_client
    async with build_async_engine_client_from_engine_args(
  File "/root/anaconda3/envs/monkey/lib/python3.9/contextlib.py", line 181, in __aenter__
    return await self.gen.__anext__()
  File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/vllm/entrypoints/openai/api_server.py", line 132, in build_async_engine_client_from_engine_args
    if (model_is_embedding(engine_args.model, engine_args.trust_remote_code,
  File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/vllm/entrypoints/openai/api_server.py", line 73, in model_is_embedding
    return ModelConfig(model=model_name,
  File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/vllm/config.py", line 227, in __init__
    self.max_model_len = _get_and_verify_max_len(
  File "/root/anaconda3/envs/monkey/lib/python3.9/site-packages/vllm/config.py", line 1740, in _get_and_verify_max_len
    assert "factor" in rope_scaling
AssertionError
```
使用top命令看了之后，一直有很多进程在编译，后来了解到vllm有很多依赖项，直接pip安装需要重新编译，不如直接使用whl文件安装

4. 其他错误
```commandline
Traceback (most recent call last):
  File "/workspace/work/Qwen-VL-master/openai_api.py", line 21, in <module>
    from transformers.generation import GenerationConfig
ModuleNotFoundError: No module named 'transformers.generation'
```
这个问题需要从源码安装Transform来解决