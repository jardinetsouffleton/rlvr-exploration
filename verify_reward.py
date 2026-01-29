
import sys
from tsp_dual_utils import calculate_dual_reward

def test_reward():
    opt = 10.0
    
    # 1. Perfect answer
    r_perfect = calculate_dual_reward(10.0, opt, overestimate_mult=20.0, overestimate_penalty=5.0)
    print(f"Perfect (10.0): {r_perfect} (Expected: 0.0)")
    assert r_perfect == 0.0
    
    # 2. Underestimate (Safe)
    # Gap = (10-9)/10 = 0.1
    # Reward = -0.1
    r_under = calculate_dual_reward(9.0, opt, overestimate_mult=20.0, overestimate_penalty=5.0)
    print(f"Under (9.0): {r_under} (Expected: -0.1)")
    assert abs(r_under - (-0.1)) < 1e-6
    
    # 3. Overestimate (Unsafe) - Small
    # Gap = (11-10)/10 = 0.1
    # Reward = -5.0 - (20.0 * 0.1) = -5.0 - 2.0 = -7.0
    r_over_small = calculate_dual_reward(11.0, opt, overestimate_mult=20.0, overestimate_penalty=5.0)
    print(f"Over Small (11.0): {r_over_small} (Expected: -7.0)")
    assert abs(r_over_small - (-7.0)) < 1e-6
    
    # 4. Critical comparison
    # Being 10% under (-0.1 reward) vs 1% over
    # Over 1%: 10.1 -> gap 0.01 -> reward = -5.0 - (20*0.01) = -5.2
    r_over_tiny = calculate_dual_reward(10.1, opt, overestimate_mult=20.0, overestimate_penalty=5.0)
    print(f"Over Tiny (10.1): {r_over_tiny}")
    
    if r_over_tiny < r_under:
        print("PASS: Tiny overestimate is strictly worse than larger underestimate")
    else:
        print("FAIL: Reward logic not strict enough")
        
    print("\nVerification Complete.")

if __name__ == "__main__":
    test_reward()
