# 基于 LangChain 的学习助手

“开源软件构建技术”第七章智能应用开发实验。使用 Python、LangChain 和 Streamlit 开发一个简单的学习辅助网页，支持知识讲解、资料摘要和自测题生成。

开源仓库：**https://github.com/xiaobing0903/langchain-study-assistant**

## 1. 实验目标与功能

实验要求是基于 LangChain，使用在线或离线大模型开发智能应用，并提交 README 和开源仓库链接。本项目对应实现如下：

| 实验要求 | 项目实现 |
| --- | --- |
| 使用 LangChain 框架 | 使用提示词模板、LCEL 调用链和文本输出解析器 |
| 接入在线或离线大模型 | 在线使用 ChatOpenAI；本地使用 ChatOllama，二选一即可 |
| 开发智能应用 | Streamlit 网页支持选择任务、输入资料、查看结果与下载 |
| README 与开源仓库 | 本文说明安装、配置、原理、测试与局限；源码按 MIT 许可开放 |

三项功能：

- **知识讲解**：输入主题，生成概念、例子、常见误区和复习建议。
- **资料摘要**：粘贴文本或上传 UTF-8 的 TXT / Markdown，提取主要内容。
- **生成自测题**：根据主题或资料生成三道选择题，并给出答案和解析。

这是单次任务应用，不包含知识库检索、多轮聊天、PDF 解析或自动评分。上传的短文本直接进入模型上下文。

## 2. 环境与安装

建议使用 **Python 3.12**。主要依赖版本记录在 `requirements.txt`，在项目专属虚拟环境中安装。

```bash
git clone https://github.com/xiaobing0903/langchain-study-assistant.git
cd langchain-study-assistant
python -m venv .venv
```

Windows PowerShell，无需激活环境即可运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

macOS / Linux：

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

浏览器访问 **http://localhost:8501**。服务默认只监听本机。若 8501 端口被占用，可在启动命令后添加 `--server.port 8502`。

## 3. 模型配置

应用需要可用的大模型才能生成结果。网页可以在没有模型配置时打开，但不会显示预制答案来冒充模型输出。

### 方式一：在线 API

在左侧选择“在线 API”，填写：

1. **模型名称**：服务商账号实际有权限调用的模型 ID。
2. **API 地址**：OpenAI 官方服务为 `https://api.openai.com/v1`；其他服务须支持标准 Chat Completions 请求，并填写其文档指定的基础地址。
3. **API Key**：在侧栏密码框中输入，不要写进源码或提交到仓库。

项目明确使用 `/chat/completions` 接口。非标准工具调用、推理字段和服务商特有参数不在支持范围内；这类需求应改用对应的 LangChain 专用集成。部分模型不支持本项目的 `temperature` 或 `max_tokens` 参数，需要按服务商文档调整 `assistant.py` 中的模型参数。

已实测的 DeepSeek 配置：模型名称填写 `deepseek-flash`，API 地址填写 `https://api.deepseek.com`，API Key 填写自己的 DeepSeek 密钥。依据 [DeepSeek 官方接口说明](https://api-docs.deepseek.com/quick_start/agent_integrations/)，本应用使用 OpenAI 兼容格式；`https://api.deepseek.com/anthropic` 属于另一种接口格式，不应填入本应用的 API 地址。

### 方式二：本地 Ollama

先按照 [Ollama 官方说明](https://github.com/ollama/ollama)安装并启动服务，下载适合电脑内存和算力的模型：

```text
ollama pull <你选择的模型名称>
ollama list
```

尖括号中的内容需要替换，不能原样运行。然后在侧栏选择“本地 Ollama”，填写 `ollama list` 中的完整模型名称，地址默认为 `http://localhost:11434`，无需 API Key。模型下载需要网络；模型准备好后，本地推理可以离线运行。本项目不自动安装 Ollama 或下载模型。

### 可选：保存本机配置

将 `.env.example` 复制为 `.env`，按其中注释填写默认值。Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

修改 `.env` 后重启应用。`.env` 已加入 `.gitignore`，不要将其上传到公开仓库。在线模式会把输入发送到你配置的模型服务；请勿上传密码或私密资料。界面生成结果仅保留在当前会话，刷新页面后可能丢失，可提前下载。

## 4. 使用与演示

1. 启动应用，配置一种可用的模型。
2. 选择“知识讲解”，输入“解释 Git 中工作区、暂存区和版本库的区别”，点击“开始生成”。
3. 选择“资料摘要”，清空输入框，上传 `examples/git_basics.txt`，查看摘要是否覆盖关键概念。
4. 使用同一份资料切换到“生成自测题”，检查是否生成三道选择题和答案解析。
5. 点击“下载 Markdown 结果”，保存学习记录。

文字输入和文件上传二选一。每次不超过 12000 个字符，文件不超过 100 KB。UTF-8 BOM 文件也可读取；扫描件、Word、PDF 和其他编码文件不支持直接上传。

结果由实际模型生成，内容可能随模型和参数变化。摘要的忠实度与题目的准确性需要人工核对；程序没有把题目格式或正确答案做成硬性校验。

## 5. 实现原理

数据流：`网页输入 → 参数与长度校验 → 提示词模板 → 大模型 → 文本解析 → 网页展示 / 下载`。

核心代码位于 `assistant.py`：

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名中文学习助手。当前任务：{instruction}"),
    ("human", "学习主题或资料：{text}"),
])
chain = prompt | model | StrOutputParser()
answer = chain.invoke({"instruction": instruction, "text": text})
```

上述片段为核心结构的简化展示，实际提示词还约束了语言、摘要范围和回答长度。三种任务使用不同指令；用户文本作为变量传入，避免文本中的花括号被当成模板变量。在线模型通过 `ChatOpenAI` 构建，本地模型通过 `ChatOllama` 构建，两者复用同一条 LangChain 链。

界面与模型逻辑分开，方便测试。在线请求设置 60 秒超时并最多重试一次，本地请求设置 120 秒超时。失败时显示排查提示，清除旧结果；不会把 SDK 原始异常和密钥显示到网页。

## 6. 项目结构

```text
langchain-study-assistant/
├── app.py                   # Streamlit 网页
├── assistant.py             # 配置校验、文件读取、LangChain 链
├── requirements.txt         # 固定主要依赖版本
├── .env.example             # 不含密钥的配置示例
├── .streamlit/config.toml   # 本机监听与界面主题
├── examples/git_basics.txt  # 可用于演示的学习资料
├── tests/test_assistant.py  # 核心逻辑、接口协议与界面测试
├── .github/workflows/tests.yml
├── README.md
└── LICENSE
```

## 7. 测试与验证边界

Windows 执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖三种任务的调用链、提示词传参、空输入和超长输入、UTF-8 文件读取、模型配置、空响应，以及网页提交、清除结果、错误处理。另有本地 HTTP 模拟服务测试，验证真实 `ChatOpenAI` 客户端发送 Chat Completions 请求并经过 LangChain 解析响应。

2026 年 11 月 3 日在 Windows、Python 3.12.14 环境验证：12 项自动化测试全部通过，`python -m pip check` 未发现依赖冲突。

**自动化测试不需要真实 API Key，测试替身仅存在于测试代码。** 仓库包含 GitHub Actions，推送后可在 Actions 页面查看测试结果。

另于 **2026 年 11 月 3 日**，通过项目原有 `create_model` 和 `run_task` 调用链，使用 DeepSeek 官方接口 `https://api.deepseek.com`、模型 `deepseek-flash` 完成三项真实模型验证：

| 功能 | 输入 | 实测结果 | 耗时 |
| --- | --- | --- | --- |
| 知识讲解 | Git 工作区、暂存区和版本库的区别 | 成功；含概念、例子、误区与复习建议 | 7.56 秒 |
| 资料摘要 | `examples/git_basics.txt` | 成功；给出主题、5 个要点和关键术语 | 12.81 秒 |
| 生成自测题 | `examples/git_basics.txt` | 成功；3 道单项选择题，每题 4 个选项，附答案和解析 | 9.25 秒 |

三次响应的 `finish_reason` 均为 `stop`，未因输出长度限制截断。人工复核后，讲解与摘要中的 Git 概念正确，自测题的答案和解析与输入资料一致。接口报告的总用量为 2608 tokens；耗时、用量和生成内容仅代表本次测试，不构成固定性能指标。

实测密钥通过隐藏终端输入，只在测试进程内存中使用，没有保存到源码、配置文件或仓库。复现时需要填写自己的有效 API Key，或使用已安装模型的 Ollama。本次未测试本地 Ollama 的真实推理。

## 8. 常见问题

| 问题 | 处理方式 |
| --- | --- |
| 找不到 streamlit 或 langchain | 确认安装和启动使用同一个 `.venv` Python |
| 提示模型名称或 API Key 缺失 | 在侧栏填写，或修改本机 `.env` 后重启 |
| 生成失败 | 核对模型权限、服务地址、网络、额度与接口参数支持情况 |
| Ollama 无法生成 | 检查服务是否启动、模型名称是否与 `ollama list` 一致 |
| 文本文件无法读取 | 另存为 UTF-8 编码的 TXT / Markdown，并检查大小限制 |
| 回答较慢 | 本地推理耗时受硬件与模型大小影响；选择适合机器的模型 |

## 9. 参考资料与许可证

- [LangChain ChatOpenAI 官方文档](https://docs.langchain.com/oss/python/integrations/chat/openai)
- [LangChain ChatOllama 官方文档](https://docs.langchain.com/oss/python/integrations/chat/ollama)
- [Streamlit 表单文档](https://docs.streamlit.io/develop/concepts/architecture/forms)

本项目源码采用 [MIT License](LICENSE)。第三方依赖和使用的模型遵循各自许可证与服务条款。

## 10. 作业提交

在课程平台上传本仓库的 **README.md**，并附上本文开头的公开仓库链接。三项功能均已通过 DeepSeek 真实调用验证；演示时仍需要在页面填写自己的模型配置。课程页面要求的交付物是 README 和仓库链接，没有要求沿用其他作业的 Word 封面或五页格式。
