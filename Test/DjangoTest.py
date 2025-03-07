import unittest
from chat import views

class LLMTest(unittest.TestCase):
    def test_get_response(self):
        chat_history = []
        response = views.get_response(chat_history, "Hello, how are you?")
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        pass