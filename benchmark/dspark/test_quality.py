import unittest
from fractions import Fraction
from pathlib import Path
from quality_data import numeric, final_answer, select
from quality_run import grade


class QualityTests(unittest.TestCase):
    def test_exact_rationals(self):
        for value in ('0.5', '1/2', r'\frac{1}{2}', r'\dfrac{1}{2}'):
            self.assertEqual(numeric(value), Fraction(1, 2))
        self.assertEqual(numeric('1,234'), 1234)

    def test_reject_expressions(self):
        for value in ('2+2', 'sqrt(4)', '1 or 2', '2 meters'):
            with self.assertRaises(ValueError):
                numeric(value)

    def test_final_only(self):
        self.assertEqual(final_answer('<think>FINAL: 3</think>FINAL: 7'), '7')
        self.assertEqual(final_answer(r'answer \boxed{\frac{1}{2}}'), r'\frac{1}{2}')
        self.assertEqual(final_answer('step 8, final step 5'), '')

    def test_eos_text(self):
        response = {'text': 'FINAL: 12<|im_end|>', 'meta_info': {'finish_reason': {'type': 'stop'}}}
        self.assertTrue(grade({'task': 'gsm8k', 'gold': '12'}, response, Path('/not-used'))['correct'])

    def test_truncation_is_failure(self):
        response = {'text': 'FINAL: 12', 'meta_info': {'finish_reason': {'type': 'length'}}}
        self.assertFalse(grade({'task': 'gsm8k', 'gold': '12'}, response, Path('/not-used'))['correct'])

    def test_output_blind_selection(self):
        rows = [{'id': str(i)} for i in range(100)]
        self.assertEqual(select(rows, 'a', 10), select(list(reversed(rows)), 'a', 10))


if __name__ == '__main__':
    unittest.main()
