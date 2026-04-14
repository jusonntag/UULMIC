import os
import sys
import psutil

class ConcurrencyManager:
    """
    Manager to dynamically evaluate machine capabilities (CPU / Memory)
    and adapt parallel worker counts safely.
    """
    
    @staticmethod
    def get_hardware_info() -> dict:
        ram = psutil.virtual_memory()
        return {
            "logical_cores": os.cpu_count() or 1,
            "total_ram_gb": ram.total / (1024 ** 3),
            "available_ram_gb": ram.available / (1024 ** 3)
        }
        
    @staticmethod
    def setup_optimal_workers(estimated_gb_per_task: float = 1.5, enforce_c_limits: bool = True) -> dict:
        """
        Calculates optimal concurrency parameters to prevent Out-Of-Memory (OOM) 
        and optionally configures process environment limitations.
        """
        hw = ConcurrencyManager.get_hardware_info()
        gil_enabled = getattr(sys, '_is_gil_enabled', lambda: True)()
        
        # Calculate bottleneck: are we CPU bound or Memory bound?
        safe_memory_workers = max(1, int(hw["available_ram_gb"] // estimated_gb_per_task))
        optimal_workers = min(hw["logical_cores"], safe_memory_workers)
        
        # Block underlying C-Level threading. If Python is already spanning X parallel 
        # threads/processes, OpenMP/MKL must be throttled to 1 thread per process.
        if enforce_c_limits:
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"
            
        return {
            "optimal_workers": optimal_workers,
            "logical_cores": hw["logical_cores"],
            "memory_bound": safe_memory_workers < hw["logical_cores"],
            "recommended_backend": "thread" if not gil_enabled else "process",
            "c_threads_enforced": enforce_c_limits,
            "gil_enabled": gil_enabled
        }

def print_diagnostics():
    print("="*40)
    print("   System Parallelism Diagnostics   ")
    print("="*40)
    
    manager = ConcurrencyManager()
    hw = manager.get_hardware_info()
    print(f"[Hardware] Total CPU Cores: {hw['logical_cores']}")
    print(f"[Hardware] Total RAM: {hw['total_ram_gb']:.2f} GB (Available: {hw['available_ram_gb']:.2f} GB)")

    # Test auto-adaptation
    config = manager.setup_optimal_workers(estimated_gb_per_task=1.5)
    
    print(f"\n[Software] Python Version: {sys.version.split(' ')[0]}")
    print(f"[Software] Global Interpreter Lock (GIL) Enabled: {config['gil_enabled']}")
    
    print("\n" + "-"*40)
    print("   Concurrency Adaptation Settings")
    print("-" * 40)
    
    # Auto-adapted results
    print(f"Optimal Workers Generated: {config['optimal_workers']} (Out of {config['logical_cores']} CPU cores)")
    if config['memory_bound']:
        print(" -> Throttled due to system memory constraints.")
    else:
        print(" -> Fully utilizing all CPU cores cleanly.")
        
    print(f"Backend Strategy Recommendation: {config['recommended_backend'].upper()}-BASED")
    print(f"Underlying Math Threading Capped (OMP/MKL): {config['c_threads_enforced']}")
    
    print("="*40)

if __name__ == '__main__':
    print_diagnostics()
