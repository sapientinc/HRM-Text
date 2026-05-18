from typing import Optional
from tqdm import tqdm
from vllm import LLM, SamplingParams

from simple_inference_engine import inference_load_checkpoint, inference_generate

class BaseEngine:
    def generate(self, prompts: list[str]) -> list[str]:
        raise NotImplementedError

class VLLMEngine(BaseEngine):
    def __init__(self, ckpt_path: str, **kwargs):
        self.llm = LLM(model=ckpt_path, **kwargs)

    def generate(self, prompts: list[str], temperature: float = 0.0, max_tokens: Optional[int] = None, stop: Optional[str | list[str]] = None) -> list[str]:
        outputs = self.llm.generate(prompts, SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop
        ))
        return [out.outputs[0].text for out in outputs]

class SimpleEngine(BaseEngine):
    def __init__(self, ckpt_path: str, ckpt_epoch: Optional[int] = None, ckpt_use_ema: bool = True):
        self.ckpt = inference_load_checkpoint(ckpt_path, ckpt_epoch, ckpt_use_ema)

    def generate(self, prompts: list[str], batch_size: int = 100, max_context: int = 1024, max_tokens: Optional[int] = None, temperature: float = 0.0, condition: str = "direct") -> list[str]:
        if max_tokens is None:
            max_tokens = max_context

        # Launch generation
        engine_prompts = [(i, (condition, p.strip())) for i, p in enumerate(prompts)]
        outputs = [""] * len(engine_prompts)

        pbar = tqdm(total=len(outputs), desc="generation")
        for gen_id, generated_text in inference_generate(
            self.ckpt, iter(engine_prompts), max_context, max_tokens, batch_size, temperature
        ):
            outputs[gen_id] = generated_text
            pbar.update()
        pbar.close()

        return outputs
