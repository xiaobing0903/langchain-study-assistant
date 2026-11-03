"""运行：python -m streamlit run app.py"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from assistant import (
    MAX_INPUT_CHARS,
    MODES,
    ModelConfig,
    create_model,
    read_text_file,
    run_task,
    validate_config,
    validate_task,
)

load_dotenv(Path(__file__).with_name(".env"))
st.set_page_config(page_title="学习助手 · LangChain", page_icon="📚", layout="centered")

with st.sidebar:
    st.header("模型设置")
    providers = ["在线 API", "本地 Ollama"]
    provider_choice = st.selectbox(
        "接入方式", providers, index=int(os.getenv("MODEL_PROVIDER") == "ollama")
    )
    provider = "openai" if provider_choice == providers[0] else "ollama"
    model_name = st.text_input("模型名称", value=os.getenv("MODEL_NAME", ""))
    if provider == "openai":
        base_url = st.text_input(
            "API 地址", value=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        api_key = st.text_input(
            "API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password"
        )
        st.caption("使用 Chat Completions 接口。请求会发送到你填写的服务地址，可能产生费用。")
    else:
        base_url = st.text_input(
            "Ollama 地址", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        api_key = ""
        st.caption("请先启动 Ollama，并下载与模型名称一致的模型。")
    if st.button("清除生成结果", key="clear_result"):
        st.session_state.pop("result", None)
    st.caption("API Key 不写入项目文件。刷新页面后可重新配置，或从本机 .env 读取。")

st.title("学习助手")
st.write("讲清一个概念，整理一段资料，再用三道题检验理解。")
st.caption("开源软件构建技术 · 第七章智能应用开发 · LangChain")

mode = st.radio("选择学习任务", list(MODES), horizontal=True)
with st.expander("试试示例"):
    st.write("主题：解释 Git 中工作区、暂存区和版本库的区别。")
    st.write("资料：仓库中的 examples/git_basics.txt 可直接上传，用于摘要和自测。")

with st.form("learning_task"):
    content = st.text_area(
        "输入主题或粘贴资料", height=180, max_chars=MAX_INPUT_CHARS,
        placeholder="例如：解释 Git 中工作区、暂存区和版本库的区别。",
    )
    upload = st.file_uploader("也可以上传 UTF-8 的 TXT / Markdown 文件", type=["txt", "md"])
    st.caption("文字和文件只选一种；每次最多 12000 个字符，文件不超过 100 KB。")
    submitted = st.form_submit_button("开始生成", type="primary", key="generate")

if submitted:
    # 新请求失败时不保留上一份答案，避免把旧结果误认为本次输出。
    st.session_state.pop("result", None)
    config = ModelConfig(provider, model_name, base_url, api_key)
    try:
        if upload is not None and content.strip():
            raise ValueError("请只使用文本框或上传文件中的一种输入方式。")
        text = read_text_file(upload.getvalue()) if upload is not None else content
        text = validate_task(mode, text)
        validate_config(config)
    except ValueError as exc:
        st.warning(str(exc))
    else:
        try:
            with st.spinner("正在通过 LangChain 调用模型…"):
                answer = run_task(mode, text, create_model(config))
            st.session_state.result = {
                "mode": mode, "model": model_name, "answer": answer,
            }
        except Exception:
            # 不把可能包含地址、请求内容或凭据的 SDK 异常原文显示出来。
            st.error(
                "生成失败。请检查模型名称、API Key、额度和网络；"
                "本地模式请确认 Ollama 已启动且模型已下载，然后重试。"
            )

if "result" in st.session_state:
    result = st.session_state.result
    st.divider()
    st.subheader(f"生成结果 · {result['mode']}")
    st.caption(f"模型：{result['model']} · 内容由模型生成，请核对关键事实。")
    st.markdown(result["answer"])
    st.download_button(
        "下载 Markdown 结果", result["answer"],
        file_name="study-result.md", mime="text/markdown",
    )
