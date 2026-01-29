"""
Tests for SAT Dual Bound Infrastructure
"""
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Import modules to test
from sat_utils import generate_sat_instance, solve_sat_backtracking, check_sat_solution
from sat_dual_utils import get_sat_partial_prompt, parse_partial_assignment, calculate_partial_assignment_reward
import sat_dual_inference

class TestSatDual(unittest.TestCase):

    def test_01_solver_fixed_assignment(self):
        print("\n=== Test 01: Backtracking Solver with Fixed Assignment ===")
        n_vars = 5
        # Generate known instance
        # x1 OR x2, -x1 OR x3 ...
        # Let's manually define clauses to be sure of answer
        # Clause 1: (1, 2) -> 1=T or 2=T
        # Clause 2: (-1, 3) -> if 1=T then 3=T
        clauses = [[1, 2], [-1, 3]] 
        
        # Solution if 1=T -> 3=T, 2=any
        # Solution if 1=F -> 2=T, 3=any
        
        # Case A: Valid Partial (1=T) -> Should find solution
        sol_a = solve_sat_backtracking(clauses, n_vars, fixed_assignment=[1])
        self.assertIsNotNone(sol_a)
        self.assertIn(1, sol_a) # Must respect fixed
        
        # Case B: Valid Partial (1=F, 2=T) -> Should find solution
        sol_b = solve_sat_backtracking(clauses, n_vars, fixed_assignment=[-1, 2])
        self.assertIsNotNone(sol_b)
        self.assertIn(-1, sol_b)
        self.assertIn(2, sol_b)
        
        # Case C: Invalid Partial (1=F, 2=F) -> Clause 1 violated immediately
        sol_c = solve_sat_backtracking(clauses, n_vars, fixed_assignment=[-1, -2])
        self.assertIsNone(sol_c)
        
        # Case D: Structurally valid but logically impossible?
        # Clause 3: (1, 2), Clause 4: (-1, -1), Clause 5: (-2, -2) -> Leads to empty? 
        # Actually easier: (1), (-1). 
        clauses_imp = [[1], [-1]]
        sol_d = solve_sat_backtracking(clauses_imp, n_vars, fixed_assignment=[])
        self.assertIsNone(sol_d)
        
        print("PASS: Solver handles fixed assignments correctly.")

    def test_02_prompt_generation(self):
        print("\n=== Test 02: Prompt Generation ===")
        clauses = [[1, -2, 3]]
        n_vars = 5
        
        prompt_cot = get_sat_partial_prompt(clauses, n_vars, mode="cot")
        self.assertIn("Find a partial boolean assignment", prompt_cot)
        self.assertIn("<thought>", prompt_cot)
        
        prompt_direct = get_sat_partial_prompt(clauses, n_vars, mode="direct")
        self.assertIn("<answer>", prompt_direct)
        self.assertNotIn("<thought>", prompt_direct)
        
        print("PASS: Prompt generation formats look correct.")

    def test_03_parsing(self):
        print("\n=== Test 03: Output Parsing ===")
        n_vars = 5
        
        # Correct format
        self.assertEqual(parse_partial_assignment("<answer>1 -2 3</answer>", n_vars), [1, -2, 3])
        
        # With thought
        txt = "<thought>foo</thought><answer>1 -5</answer>"
        self.assertEqual(parse_partial_assignment(txt, n_vars), [1, -5])
        
        # Noise
        txt = "Sure! <answer>1 2</answer> done."
        self.assertEqual(parse_partial_assignment(txt, n_vars), [1, 2])
        
        # Out of bounds
        txt = "<answer>1 6 -2</answer>" # 6 > 5
        self.assertEqual(parse_partial_assignment(txt, n_vars), [1, -2])
        
        # Empty
        txt = "<answer></answer>"
        self.assertEqual(parse_partial_assignment(txt, n_vars), [])
        
        print("PASS: Parsing is robust.")

    def test_04_reward_logic(self):
        print("\n=== Test 04: Reward Calculation ===")
        n_vars = 3
        clauses = [[1, 2], [-1, 3]] 
        # Solutions: (1, 3, 2), (1, 3, -2), (-1, 2, 3), (-1, 2, -3)
        
        # Valid partial: [1] -> matches first two
        r_valid = calculate_partial_assignment_reward([1], clauses, n_vars)
        self.assertGreater(r_valid, 0)
        self.assertAlmostEqual(r_valid, 1/3)
        
        # Valid partial bigger: [1, 3]
        r_valid_2 = calculate_partial_assignment_reward([1, 3], clauses, n_vars)
        self.assertAlmostEqual(r_valid_2, 2/3)
        
        # Invalid partial: [-1, -2] -> violates clause 1
        r_invalid = calculate_partial_assignment_reward([-1, -2], clauses, n_vars)
        self.assertLess(r_invalid, 0)
        self.assertEqual(r_invalid, -5.0) # Default penalty
        
        print("PASS: Reward logic assigns positive/negative correctly.")

    @patch("sat_dual_inference.AutoTokenizer")
    @patch("sat_dual_inference.AutoModelForCausalLM")
    def test_05_inference_flow(self, mock_model_cls, mock_tokenizer_cls):
        print("\n=== Test 05: Mock Inference Flow ===")
        
        # Setup Mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Mock Tokenizer
        mock_tokenizer.apply_chat_template.return_value = "PROMPT"
        # inputs needs to be a dict of tensors (mocks that have .to())
        mock_input = MagicMock()
        mock_input.to.return_value = mock_input # .to returns self
        mock_tokenizer.return_value = {"input_ids": mock_input, "attention_mask": mock_input}
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.decode.return_value = "<answer>1 -2</answer>"
        
        # Mock Generate
        # We don't care about actual tensor out, just that decode gets called
        # generate returns a tensor (mock) that we then slice
        mock_output = MagicMock()
        # Slicing returns self
        mock_output.__getitem__.return_value = mock_output 
        mock_model.generate.return_value = mock_output
        mock_model.device = "cpu"
        
        clauses = [[1, 2]]
        n_vars = 5
        
        # Run Predict
        result = sat_dual_inference.predict_partial_assignment(mock_model, mock_tokenizer, clauses, n_vars)
        
        self.assertEqual(result, [1, -2])
        print("PASS: Inference pipeline connects correctly.")

if __name__ == "__main__":
    unittest.main()
