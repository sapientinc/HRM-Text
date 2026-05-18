from typing import Any, Optional
from collections import defaultdict
import pydantic
import json
from omegaconf import OmegaConf

from utils.functions import load_model_class


class BenchmarkConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    generation_config: dict[str, Any] = {}


class EvaluationConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    run_only: Optional[list[str]] = None
    engine: str
    generation_config: dict[str, Any] = {}
    benchmarks: list[BenchmarkConfig]

    @pydantic.model_validator(mode='after')
    def check_run_only_against_benchmarks(self):
        if self.run_only is not None:
            assert self.run_only, "run_only cannot be empty."

            valid_set = {b_cfg.name for b_cfg in self.benchmarks}
            for b_name in self.run_only:
                if b_name not in valid_set:
                    raise ValueError(f"Unknown benchmark name in run_only: {b_name}")

        return self


def main():
    # 1. Load and Override Config
    cli_conf = OmegaConf.from_cli()
    config_path = cli_conf.pop("config", "evaluation/config/hrm_benchmarking.yaml")
    # Merge YAML config with CLI overrides
    base_conf = OmegaConf.load(config_path)
    cfg = EvaluationConfig(**OmegaConf.to_container(OmegaConf.merge(base_conf, cli_conf), resolve=True))  # type: ignore

    # 2. Initialize Engine
    print(f"Initializing Engine: {cfg.engine}...")
    engine_cls = load_model_class(f"engines@{cfg.engine}", prefix="evaluation.")
    engine = engine_cls(**(cfg.__pydantic_extra__ or {}))

    # 3. Group Benchmarks by Generation Config
    # To minimize bubbles, we group benchmarks sharing the exact same generation kwargs
    print("Preparing and grouping benchmarks...")

    # grouped_tasks maps a hashed config tuple -> {"prompts": [...], "benchmarks": [...]}
    grouped_tasks = defaultdict(lambda: {"prompts": [], "benchmarks": []})
    for b_cfg in cfg.benchmarks:
        b_name = b_cfg.name
        if cfg.run_only is not None and b_name not in cfg.run_only:
            continue

        # Instantiate benchmark
        bench_cls = load_model_class(f"benchmarks@{b_name}", prefix="evaluation.")
        benchmark = bench_cls(**(b_cfg.__pydantic_extra__ or {}))

        # Resolve final generation config: Base -> Benchmark Specific -> Benchmark Overrides
        gen_cfg = cfg.generation_config | benchmark.generation_overrides | b_cfg.generation_config
        # Hash key for grouping identical ones
        gen_key = json.dumps(gen_cfg, sort_keys=True)
        
        # Track offsets to map flattened generations back to their source benchmarks
        start_idx = len(grouped_tasks[gen_key]["prompts"])
        grouped_tasks[gen_key]["prompts"].extend(benchmark.prompts)
        end_idx = len(grouped_tasks[gen_key]["prompts"])
        
        grouped_tasks[gen_key]["benchmarks"].append((b_name, benchmark, start_idx, end_idx))

    # 4. Generate and Evaluate per Group
    all_results = {}
    
    for gen_key, group in grouped_tasks.items():
        gen_kwargs = json.loads(gen_key)
        prompt_template = gen_kwargs.pop("prompt_template", "{prompt}")
        # Apply prompt templates
        prompts = [prompt_template.format(prompt=s) for s in group["prompts"]]
        
        print("\n" + "="*50)
        print(f"Running generation batch (Size: {len(prompts)}) with config:")
        for k, v in gen_kwargs.items():
            print(f"  {k}: {v}")
        print("="*50)

        # Generate all prompts for this config group
        generations = engine.generate(prompts, **gen_kwargs)

        # Dispatch results back to individual benchmarks
        for b_name, benchmark, start_idx, end_idx in group["benchmarks"]:
            b_generations = generations[start_idx:end_idx]
            metrics = benchmark.compute_metrics(b_generations)
            all_results[b_name] = metrics

    # 5. Summary Report
    print("\n" + "#"*50 + "\nEVALUATION SUMMARY\n" + "#"*50)
    for b_name, metrics in all_results.items():
        print(f"\n--- {b_name} ---")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k:.<25}: {v:.4f}")
            else:
                print(f"{k:.<25}: {v}")

if __name__ == "__main__":
    main()
