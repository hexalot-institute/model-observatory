import unittest

from model_observatory import canonicalize_benchmark

class TestBenchmarkCanonicalization(unittest.TestCase):
    def test_tier1_echo_punctuation_tolerant(self):
        ok_a, a = canonicalize_benchmark(prompt_index=0, response_text="OK")
        ok_b, b = canonicalize_benchmark(prompt_index=0, response_text="  ok.  \n")
        self.assertTrue(ok_a)
        self.assertTrue(ok_b)
        self.assertEqual(a, b)

    def test_tier1_arithmetic_tolerant(self):
        ok_a, a = canonicalize_benchmark(prompt_index=1, response_text="4")
        ok_b, b = canonicalize_benchmark(prompt_index=1, response_text="2+2 = 4.")
        self.assertTrue(ok_a)
        self.assertTrue(ok_b)
        self.assertEqual(a, b)

    def test_tier1_sequence(self):
        ok_a, a = canonicalize_benchmark(prompt_index=2, response_text="8")
        ok_b, b = canonicalize_benchmark(prompt_index=2, response_text="The next number is 8.")
        self.assertTrue(ok_a)
        self.assertTrue(ok_b)
        self.assertEqual(a, b)

    def test_tier3_json_canonicalization(self):
        ok_a, a = canonicalize_benchmark(prompt_index=4, response_text='{"a":1,"b":2}')
        ok_b, b = canonicalize_benchmark(prompt_index=4, response_text='{\n  "b": 2,\n  "a": 1\n}')
        self.assertTrue(ok_a)
        self.assertTrue(ok_b)
        self.assertEqual(a, b)

    def test_tier4_tool_call_whitespace_tolerant(self):
        ok_a, a = canonicalize_benchmark(prompt_index=5, response_text='get_weather(city="Tokyo", units="celsius")')
        ok_b, b = canonicalize_benchmark(prompt_index=5, response_text='get_weather(city="Tokyo", units="celsius")\n')
        self.assertTrue(ok_a)
        self.assertTrue(ok_b)
        self.assertEqual(a, b)

    def test_empty_response_fails(self):
        ok, val = canonicalize_benchmark(prompt_index=0, response_text="")
        self.assertFalse(ok)

    def test_whitespace_only_fails(self):
        ok, val = canonicalize_benchmark(prompt_index=1, response_text="   \n  ")
        self.assertFalse(ok)

if __name__ == "__main__":
    unittest.main()
