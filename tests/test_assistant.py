import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from streamlit.testing.v1 import AppTest

from assistant import (
    MAX_INPUT_CHARS, MODES, ModelConfig, create_model, read_text_file,
    run_task, validate_config, validate_task,
)


class CoreTests(unittest.TestCase):
    def test_all_modes_use_the_chain(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                self.assertEqual(run_task(mode, "解释 Git", FakeListChatModel(responses=["测试答案"])), "测试答案")

    def test_prompt_preserves_user_data_and_task(self):
        captured = []
        def inspect(prompt):
            captured.append(prompt.to_messages())
            return "done"
        run_task("资料摘要", '{"topic": "Git"}', RunnableLambda(inspect))
        self.assertIn("不添加原文没有的事实", captured[0][0].content)
        self.assertIn('{"topic": "Git"}', captured[0][1].content)

    def test_invalid_inputs(self):
        for mode, text in [("未知", "Git"), ("知识讲解", "  "), ("资料摘要", "字" * (MAX_INPUT_CHARS + 1))]:
            with self.assertRaises(ValueError):
                validate_task(mode, text)

    def test_empty_model_output_is_failure(self):
        with self.assertRaises(RuntimeError):
            run_task("知识讲解", "Git", FakeListChatModel(responses=[" "]))

    def test_file_reading(self):
        self.assertEqual(read_text_file("Git 资料".encode("utf-8-sig")), "Git 资料")
        for data in [b"", b"\xff", b"hello\x00", b"a" * 100001, "字".encode() * 12001]:
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                read_text_file(data)

    def test_config_validation_and_secret_repr(self):
        good = ModelConfig("openai", "test-model", "https://api.example.com/v1", "test-secret")
        validate_config(good)
        self.assertNotIn("test-secret", repr(good))
        for config in [
            ModelConfig("bad", "model", "https://example.com", "key"),
            ModelConfig("openai", "", "https://example.com", "key"),
            ModelConfig("openai", "model", "https://example.com", ""),
            ModelConfig("openai", "model", "http://example.com", "key"),
            ModelConfig("openai", "model", "https://u:p@example.com", "key"),
            ModelConfig("openai", "model", "https://example.com?token=secret", "key"),
        ]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)

    def test_ollama_needs_no_api_key(self):
        config = ModelConfig("ollama", "test-model", "http://localhost:11434")
        validate_config(config)
        self.assertEqual(create_model(config).model, "test-model")

    def test_chat_openai_http_roundtrip(self):
        # 本地模拟服务只验证协议与链路，不代表真实大模型的回答质量。
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                body = json.dumps({
                    "id": "test", "object": "chat.completion", "created": 1, "model": "test-model",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "模拟服务返回成功"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = create_model(ModelConfig("openai", "test-model", f"http://127.0.0.1:{server.server_port}/v1", "test-only"))
            self.assertEqual(run_task("知识讲解", "Git", model), "模拟服务返回成功")
            self.assertEqual(seen[0][0], "/v1/chat/completions")
            self.assertEqual(seen[0][1]["model"], "test-model")
            self.assertIn("Git", seen[0][1]["messages"][1]["content"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()

    def test_page_loads_and_empty_input_warns(self):
        self.assertFalse(self.app.exception)
        self.app.button(key="generate").click().run()
        self.assertIn("请先输入", self.app.warning[0].value)

    def test_missing_model_warns(self):
        self.app.text_input[0].input("")
        self.app.text_area[0].input("Git")
        self.app.button(key="generate").click().run()
        self.assertIn("模型名称", self.app.warning[0].value)

    def test_success_and_clear(self):
        self.app.text_input[0].input("test-model")
        self.app.text_input[2].input("test-only")
        self.app.text_area[0].input("Git")
        with patch("assistant.create_model", return_value=FakeListChatModel(responses=["测试生成结果"])):
            self.app.button(key="generate").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["result"]["answer"], "测试生成结果")
        self.app.button(key="clear_result").click().run()
        self.assertNotIn("result", self.app.session_state)

    def test_sdk_error_does_not_leak_secret_or_old_result(self):
        self.app.text_input[0].input("test-model")
        self.app.text_input[2].input("test-only")
        self.app.text_area[0].input("Git")
        self.app.session_state["result"] = {"mode": "知识讲解", "model": "old", "answer": "old-answer"}
        with patch("assistant.create_model", side_effect=RuntimeError("secret-test-token")):
            self.app.button(key="generate").click().run()
        self.assertNotIn("secret-test-token", self.app.error[0].value)
        self.assertNotIn("result", self.app.session_state)
        self.assertFalse(self.app.exception)


if __name__ == "__main__":
    unittest.main()
