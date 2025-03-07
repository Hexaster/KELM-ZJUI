import unittest
from LLM import LLM_loader
# This file tests everything about the LLM_loader class

class TestLLMLoader(unittest.TestCase):
    def test_create_loader(self):
        llm = LLM_loader.LLM_loader()
        self.assertIsNotNone(llm)
        self.assertIsInstance(llm, LLM_loader.LLM_loader)
        self.assertIsNotNone(llm.tokeniser)
        self.assertIsInstance(llm.messages, list)
        self.assertEqual(llm.default_system_prompt, {
        "role": "system",
        "content": "You are KELM, a knowledge-enhanced language model that can help you answer questions about your knowledge base."
    })
        pass

    def test_set_messages(self):
        llm = LLM_loader.LLM_loader()
        self.assertEqual(len(llm.messages), 1)
        self.assertEqual(llm.messages[0]["role"], f"{llm.default_system_prompt['role']}")
        self.assertEqual(llm.messages[0]["content"], f"{llm.default_system_prompt['content']}")
        llm.set_messages([{"role": "user", "content": "Hello, how are you?"},
                          {"role": "assistant", "content": "I'm fine, thank you."}])
        self.assertEqual(len(llm.messages), 3)
        self.assertEqual(llm.messages[1]["role"], "user")
        self.assertEqual(llm.messages[1]["content"], "Hello, how are you?")
        self.assertEqual(llm.messages[2]["role"], "assistant")
        self.assertEqual(llm.messages[2]["content"], "I'm fine, thank you.")
        pass

    def test_apply_chat_template(self):
        llm = LLM_loader.LLM_loader()
        llm.set_messages([{"role": "user", "content": "Hello, how are you?"}])
        self.assertEqual(llm.apply_chat_template(), f"System: {llm.default_system_prompt['content']}\n"
                                                    "User: Hello, how are you?\n"
                                                    "Assistant: [NEW_RESPONSE]")
        pass

    def test_generate_response(self):
        llm = LLM_loader.LLM_loader()
        prompt = "Hello, how are you?"
        generated_text = llm.generate_response(prompt)
        self.assertIsNotNone(generated_text) # At least it replies sth
        # The llm.messages should update 2 messages: one from the user and one from the model
        self.assertEqual(len(llm.messages), 3)
