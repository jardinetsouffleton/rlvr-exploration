"""
Background SAT Instance Generator

Generates verified-solvable SAT instances in a background process,
storing them in a shared queue for the training loop to consume.
"""

import multiprocessing as mp
from multiprocessing import Process, Queue
import time
import logging
from typing import Optional
from collections import defaultdict

from sat_utils import generate_sat_instance, solve_sat_backtracking

logger = logging.getLogger(__name__)


class SolvableInstanceGenerator:
    """
    Background process that generates solvable SAT instances.
    
    Maintains a buffer of pre-verified instances for each problem size,
    continuously replenishing as instances are consumed.
    """
    
    def __init__(
        self,
        min_vars: int = 5,
        max_vars: int = 20,
        clause_ratio: float = 4.2,
        buffer_size: int = 50,
        num_workers: int = 2
    ):
        """
        Args:
            min_vars: Minimum number of variables to generate
            max_vars: Maximum number of variables to generate
            clause_ratio: Clauses-per-variable ratio (4.2 is 3-SAT phase transition)
            buffer_size: Target number of instances to maintain per size
            num_workers: Number of background worker processes
        """
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.clause_ratio = clause_ratio
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        
        # Shared queue for instances (keyed by n_vars in the instance dict)
        self.instance_queue = Queue(maxsize=buffer_size * (max_vars - min_vars + 1))
        
        # Control flags
        self.stop_event = mp.Event()
        self.workers = []
        
    def start(self):
        """Start background worker processes."""
        logger.info(f"Starting {self.num_workers} instance generator workers...")
        for i in range(self.num_workers):
            p = Process(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            p.start()
            self.workers.append(p)
        logger.info("Instance generator started.")
    
    def stop(self):
        """Stop all background workers."""
        logger.info("Stopping instance generator...")
        self.stop_event.set()
        for p in self.workers:
            p.terminate()
            p.join(timeout=2)
        self.workers.clear()
        logger.info("Instance generator stopped.")
    
    def _worker_loop(self, worker_id: int):
        """Worker loop that continuously generates solvable instances."""
        import random
        import numpy as np
        
        # Seed each worker differently
        seed = int(time.time() * 1000) + worker_id
        random.seed(seed)
        np.random.seed(seed % (2**32))
        
        while not self.stop_event.is_set():
            # Cycle through all sizes
            for n_vars in range(self.min_vars, self.max_vars + 1):
                if self.stop_event.is_set():
                    break
                    
                n_clauses = int(n_vars * self.clause_ratio)
                
                # Generate and verify
                instance = self._generate_solvable_instance(n_vars, n_clauses)
                if instance is not None:
                    try:
                        self.instance_queue.put(instance, timeout=1)
                    except:
                        # Queue full, skip
                        pass
    
    def _generate_solvable_instance(self, n_vars: int, n_clauses: int, max_attempts: int = 100):
        """
        Generate a single solvable SAT instance.
        
        Returns dict with 'clauses', 'n_vars', 'n_clauses', 'solution' or None if failed.
        """
        for _ in range(max_attempts):
            clauses = generate_sat_instance(n_vars=n_vars, n_clauses=n_clauses)
            solution = solve_sat_backtracking(clauses, n_vars)
            
            if solution is not None:
                return {
                    'clauses': clauses,
                    'n_vars': n_vars,
                    'n_clauses': n_clauses,
                    'solution': solution
                }
        
        # Failed to generate solvable instance (unlikely with reasonable ratios)
        logger.warning(f"Failed to generate solvable instance for n_vars={n_vars}")
        return None
    
    @classmethod
    def test_solvability(cls, n_tests: int = 100, n_vars: int = 10, clause_ratio: float = 4.2):
        """Test that generated instances are indeed solvable."""
        generator = cls(min_vars=n_vars, max_vars=n_vars, clause_ratio=clause_ratio)
        n_clauses = int(n_vars * clause_ratio)
        
        success = 0
        for _ in range(n_tests):
            instance = generator._generate_solvable_instance(n_vars, n_clauses)
            if instance is not None:
                # Double-check solution
                from sat_utils import check_sat_solution
                is_sat, _ = check_sat_solution(instance['clauses'], instance['solution'])
                if is_sat:
                    success += 1
        
        print(f"Solvability test: {success}/{n_tests} instances verified solvable")
        assert success == n_tests, f"Only {success}/{n_tests} instances were solvable!"
        return True


class InstancePool:
    """
    Thread-safe pool for drawing instances during training.
    
    Wraps the generator queue and provides blocking get() by size.
    """
    
    def __init__(self, generator: SolvableInstanceGenerator):
        self.generator = generator
        # Local buffer organized by size
        self._buffers: dict[int, list] = defaultdict(list)
        self._lock = mp.Lock() if mp.current_process().name == 'MainProcess' else None
    
    def get_instance(self, n_vars: int, timeout: float = 30.0) -> Optional[dict]:
        """
        Get a solvable instance of the specified size.
        
        Blocks until an instance is available or timeout.
        """
        # First check local buffer
        if self._buffers[n_vars]:
            return self._buffers[n_vars].pop()
        
        # Draw from queue until we get one of the right size
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                instance = self.generator.instance_queue.get(timeout=1.0)
                inst_size = instance['n_vars']
                
                if inst_size == n_vars:
                    return instance
                else:
                    # Buffer for later
                    self._buffers[inst_size].append(instance)
            except:
                continue
        
        # Timeout - generate synchronously as fallback
        logger.warning(f"InstancePool timeout for n_vars={n_vars}, generating synchronously")
        n_clauses = int(n_vars * self.generator.clause_ratio)
        return self.generator._generate_solvable_instance(n_vars, n_clauses)
    
    def prefill(self, sizes: list[int], count_per_size: int = 10):
        """Pre-fill buffers for specified sizes."""
        logger.info(f"Pre-filling instance pool for sizes {sizes}...")
        for size in sizes:
            for _ in range(count_per_size):
                instance = self.get_instance(size, timeout=5.0)
                if instance:
                    self._buffers[size].append(instance)
        logger.info("Instance pool pre-filled.")


# Module-level singleton for easy import
_generator: Optional[SolvableInstanceGenerator] = None
_pool: Optional[InstancePool] = None


def init_instance_pool(
    min_vars: int = 5,
    max_vars: int = 20,
    clause_ratio: float = 4.2,
    buffer_size: int = 50,
    num_workers: int = 2
) -> InstancePool:
    """Initialize and start the global instance pool."""
    global _generator, _pool
    
    if _generator is not None:
        _generator.stop()
    
    _generator = SolvableInstanceGenerator(
        min_vars=min_vars,
        max_vars=max_vars,
        clause_ratio=clause_ratio,
        buffer_size=buffer_size,
        num_workers=num_workers
    )
    _generator.start()
    _pool = InstancePool(_generator)
    
    return _pool


def get_instance_pool() -> Optional[InstancePool]:
    """Get the global instance pool (must call init_instance_pool first)."""
    return _pool


def shutdown_instance_pool():
    """Shutdown the global instance pool."""
    global _generator, _pool
    if _generator is not None:
        _generator.stop()
        _generator = None
        _pool = None


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Testing SolvableInstanceGenerator...")
    SolvableInstanceGenerator.test_solvability(50, n_vars=10)
    
    print("\nTesting InstancePool...")
    pool = init_instance_pool(min_vars=5, max_vars=10)
    time.sleep(2)  # Let workers generate some instances
    
    for size in [5, 7, 10]:
        instance = pool.get_instance(size)
        if instance:
            print(f"Got instance: n_vars={instance['n_vars']}, n_clauses={instance['n_clauses']}")
        else:
            print(f"Failed to get instance for size {size}")
    
    shutdown_instance_pool()
    print("Done!")
