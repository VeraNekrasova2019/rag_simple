import os
import unittest
from rag.rag_simple import SimpleRAG

API_KEY = 'OPENAI_API_KEY'


class MyTestCase(unittest.TestCase):
    def test_simple_rag(self):
        api_key = os.getenv(API_KEY)
        if not api_key:
            raise ValueError("API key for OpenAI is not set. Please set the 'OPENAI_API_KEY' environment variable.")

        simple_rag = SimpleRAG(api_key)
        response = simple_rag.build_simple_rag()

        self.assertIsNotNone(response)
