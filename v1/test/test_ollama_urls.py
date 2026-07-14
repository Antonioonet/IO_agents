import unittest

from ollama_urls import ollama_native_url, ollama_openai_url


class OllamaUrlTests(unittest.TestCase):
    def test_derives_both_api_bases_from_a_native_root(self):
        root = "http://127.0.0.1:11434/"

        self.assertEqual(ollama_native_url(root), "http://127.0.0.1:11434")
        self.assertEqual(
            ollama_openai_url(root),
            "http://127.0.0.1:11434/v1",
        )

    def test_accepts_an_existing_openai_compatible_url(self):
        openai_url = "http://127.0.0.1:11434/v1/"

        self.assertEqual(
            ollama_native_url(openai_url),
            "http://127.0.0.1:11434",
        )
        self.assertEqual(
            ollama_openai_url(openai_url),
            "http://127.0.0.1:11434/v1",
        )


if __name__ == "__main__":
    unittest.main()
