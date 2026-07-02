from typing import Any, Optional, Callable
import re
import ast
import json
import subprocess
import tempfile
import os
from collections import defaultdict, Counter
from dataclasses import dataclass

from datasets import load_dataset, get_dataset_config_names
from math_verify import parse, verify
from lm_eval.tasks.drop.utils import process_results as drop_process_results, process_docs as drop_process_docs

from utils.functions import last_boxed_only_string, compute_benchmark_micro_macro_avg

class BaseBenchmark:
    def __init__(self):
        self.prompts = []
        self.ground_truths = []

    @property
    def generation_overrides(self) -> dict:
        """Override default generation settings (e.g., max_tokens)."""
        return {}

    def compute_metrics(self, generations: list[str]) -> dict:
        raise NotImplementedError

# --- Mathematical Reasoning ---

class GSM8k(BaseBenchmark):
    def __init__(self, split: str = "test"):
        super().__init__()

        dataset = load_dataset("gsm8k", "main", split=split)
        self.prompts = dataset["question"]
        self.ground_truths = [self._extract_truth(sol) for sol in dataset["answer"]]

    def _extract_truth(self, sol: str):
        return int(sol.split("####")[-1].strip().replace(",", ""))

    def _extract_answer(self, text: str):
        boxed = last_boxed_only_string(text)
        if boxed:
            text = boxed
        # Remove commas, $ and (e.g., 1,200 -> 1200) and whitespace
        text = text.replace(",", "").replace("$", "").strip()
        try:
            return int(float(text))
        except (ValueError, OverflowError):
            return None

    def compute_metrics(self, generations: list[str]) -> dict:
        correct, invalid, total = 0, 0, len(generations)

        for i, text in enumerate(generations):
            ans = self._extract_answer(text)
            if ans is None:
                invalid += 1
            elif ans == self.ground_truths[i]:
                correct += 1

        return {
            "n": total,
            "acc": correct / max(1, total),
            "invalid": invalid / max(1, total)
        }

class MATH(BaseBenchmark):
    def __init__(self, split: str = "test"):
        super().__init__()

        for subset in get_dataset_config_names("EleutherAI/hendrycks_math"):
            dataset = load_dataset("EleutherAI/hendrycks_math", subset, split=split)
            for item in dataset:
                label = last_boxed_only_string(item["solution"])  # type: ignore
                assert label is not None

                self.prompts.append(item["problem"])  # type: ignore
                self.ground_truths.append(label)

    def compute_metrics(self, generations: list[str]) -> dict:
        correct, invalid, total = 0, 0, len(generations)

        for i, text in enumerate(generations):
            ans = last_boxed_only_string(text)
            if ans is None:
                invalid += 1
                ans = text
            if verify(parse(self.ground_truths[i]), parse(ans)):
                correct += 1

        return {
            "n": total,
            "acc": correct / max(1, total),
            "invalid": invalid / max(1, total)
        }

# --- Reading Comprehension ---

class DROP(BaseBenchmark):
    EXAMPLES = [
        "To start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The second half was relatively quiet, with each team only scoring one touchdown. First, Detroit's Calvin Johnson caught a 1-yard pass in the third quarter. The game's final points came when Mike Williams of Tampa Bay caught a 5-yard pass. The Lions won their regular season opener for the first time since 2007\nQ: How many points did the buccaneers need to tie in the first?\nA: 3",
        "Trying to snap a two-game skid, the Bills flew to Gillette Stadium for a Week 3 divisional fight with the New England Patriots. In the first quarter, QB J. P. Losman was immediately injured on the first offensive play of the game. He would finish the series, but ended up on the bench for the rest of the game. After New England took the lead with kicker Stephen Gostkowski's 24-yard field goal, rookie QB Trent Edwards played the rest of the game for Buffalo. The Bills would get their only score of the game as RB Marshawn Lynch got an 8-yard TD run, and a Rian Lindell extra point put the Bills ahead surprisingly 7-3. However, in the second quarter, the Patriots were able to open up their running game when Bills rookie standout Paul Posluszny was lost due to a broken arm. This left passing lanes open, and for the rest of the game, the Patriots dominated. QB Tom Brady's 8-yard TD pass to TE Benjamin Watson and a 3-yard TD pass to WR Randy Moss made it 17-7 at the half. In the third quarter, New England continued its conquest with Brady's 4-yard TD pass to WR Jabar Gaffney and RB Sammy Morris' 4-yard TD run. In the fourth quarter, the Patriots ended the day with Brady and Moss hooking up with each other again on a 45-yard TD pass.\nQ: How many games had the Bills won before this game?\nA: 0",
        "The French king, John II, had been held captive in England. The Treaty of Brétigny set his ransom at 3 million crowns and allowed for hostages to be held in lieu of John. The hostages included two of his sons, several princes and nobles, four inhabitants of Paris, and two citizens from each of the nineteen principal towns of France. While these hostages were held, John returned to France to try and raise funds to pay the ransom. In 1362 John's son Louis of Anjou, a hostage in English-held Calais, escaped captivity. So, with his stand-in hostage gone, John felt honor-bound to return to captivity in England. The French crown had been at odds with Navarre since 1354, and in 1363 the Navarrese used the captivity of John II in London and the political weakness of the Dauphin to try to seize power. Although there was no formal treaty, Edward III supported the Navarrese moves, particularly as there was a prospect that he might gain control over the northern and western provinces as a consequence. With this in mind, Edward deliberately slowed the peace negotiations. In 1364, John II died in London, while still in honourable captivity. Charles V succeeded him as king of France. On 7 May 1364, one month after the dauphin's accession and three days before his coronation as Charles V, the Navarrese suffered a crushing defeat at the Battle of Cocherel.\nQ: How many years before Navarrase used the captivity of John II?\nA: 9",
    ]

    def __init__(self, split: str = "validation", num_shots: int = 3):
        super().__init__()
        # Build the few-shot prefix
        assert 0 <= num_shots <= len(self.EXAMPLES), f"num_shots must be between 0 and {len(self.EXAMPLES)}"
        prefix = "".join(f"{shot}\n\n" for shot in self.EXAMPLES[:num_shots])

        dataset = load_dataset("EleutherAI/drop", split=split)

        self.ground_truths: list[dict[str, Any]] = list(drop_process_docs(dataset))  # pyright: ignore[reportAttributeAccessIssue]
        self.prompts = [
            f"{prefix}{doc['passage']}\nQ: {doc['question']}\nA:" 
            for doc in self.ground_truths
        ]

    def compute_metrics(self, generations: list[str]) -> dict:
        total_em, total_f1, total = 0.0, 0.0, len(generations)

        for i, text in enumerate(generations):
            metrics = drop_process_results(self.ground_truths[i], [text.strip()])
            total_em += metrics.get("em", 0.0)
            total_f1 += metrics.get("f1", 0.0)
            
        return {
            "n": total,
            "em": total_em / max(1, total),
            "f1": total_f1 / max(1, total)
        }

# --- MMLU-Pro ---

class MMLUPro(BaseBenchmark):
    def __init__(self, split: str = "test"):
        super().__init__()
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", "default", split=split)
        for item in dataset:
            prompt = item["question"] + "\nOptions:\n"  # type: ignore
            current_options = []
            for idx, opt in enumerate(item["options"]):  # type: ignore
                letter = chr(65 + idx)
                prompt += f"({letter}) {opt}\n"
                current_options.append(letter)
            
            self.prompts.append(prompt)
            self.ground_truths.append((item["answer"], current_options, item["category"]))  # type: ignore

    def _extract_letter(self, text: str, options: list[str]):
        # 1. Look for options with brackets: (A), (B), etc.
        pattern_bracket = r'\((' + '|'.join(options) + r')\)'
        matches_bracket = re.findall(pattern_bracket, text)
        if matches_bracket:
            return matches_bracket[-1]
        
        # 2. If not found, match last valid option letter (A, B, C...) standing alone
        pattern_plain = r'\b(' + '|'.join(options) + r')\b'
        matches_plain = re.findall(pattern_plain, text)
        if matches_plain:
            return matches_plain[-1]
        return None

    def compute_metrics(self, generations: list[str]) -> dict:
        stats = defaultdict(lambda: {"n": 0, "correct": 0.0, "invalid": 0})
        
        for i, text in enumerate(generations):
            gt, options, subset = self.ground_truths[i]

            boxed = last_boxed_only_string(text)
            if boxed:
                text = boxed
            ans = self._extract_letter(text, options)
            
            stats[subset]["n"] += 1
            if ans is None:
                stats[subset]["invalid"] += 1
                stats[subset]["correct"] += 1 / len(options)
            elif ans == gt:
                stats[subset]["correct"] += 1
        
        return compute_benchmark_micro_macro_avg(stats)

# --- Standard Multiple Choice ---

@dataclass
class MCQDoc:
    query: str
    choices: list[str]
    gold_index: int

def _format_mcq(doc: MCQDoc, include_gold: bool) -> tuple[str, list[str]]:
    text = doc.query.strip() + "\n"
    choices = []
    for idx, choice in enumerate(doc.choices):
        letter = chr(ord('A') + idx)
        text += f"{letter}. {choice.strip()}\n"
        choices.append(letter)
    
    text += f"Answer: {chr(ord('A') + doc.gold_index)}" if include_gold else "Answer:"
    return text, choices

def _construct_mcq_prompt(target: MCQDoc, shots: list[MCQDoc]) -> tuple[str, list[str]]:
    prompt = ""
    for shot in shots:
        shot_text, _ = _format_mcq(shot, include_gold=True)
        prompt += shot_text + "\n\n"
    target_text, choices = _format_mcq(target, include_gold=False)
    return prompt + target_text, choices

class StandardMCQBenchmark(BaseBenchmark):
    @property
    def generation_overrides(self) -> dict:
        return {"max_tokens": 1}

    def compute_metrics(self, generations: list[str]) -> dict:
        correct, invalid, total = 0.0, 0, len(generations)
        
        for pred, gt in zip(generations, self.ground_truths):
            pred_clean = pred.strip().upper()
            if pred_clean not in gt["valid_set"]:
                invalid += 1
                correct += 1 / len(gt["valid_set"])
            elif pred_clean == gt["gold"]:
                correct += 1

        return {
            "n": total,
            "acc": correct / max(1, total),
            "invalid": invalid / max(1, total)
        }

class MMLU(StandardMCQBenchmark):
    def __init__(self, split: str = "test", num_shots: int = 5, special_shots: Optional[dict] = None):
        super().__init__()
        special_shots = special_shots or {}

        # Load dev shots
        shot_ds = load_dataset("cais/mmlu", "all", split="dev")
        shot_docs = defaultdict(lambda: [])
        for row in shot_ds:
            shot_docs[row["subject"]].append(MCQDoc(row['question'], row['choices'], row['answer']))  # type: ignore

        # Load test
        ds = load_dataset("cais/mmlu", "all", split=split)
        for row in ds:
            subject: str = row["subject"]  # type: ignore
            doc = MCQDoc(row['question'], row['choices'], row['answer'])  # type: ignore

            prompt, choices = _construct_mcq_prompt(doc, shot_docs[subject][:special_shots.get(subject, num_shots)])
            self.prompts.append(prompt)
            self.ground_truths.append({
                "valid_set": set(choices),
                "gold": chr(ord('A') + doc.gold_index),
                "subject": subject
            })

    def compute_metrics(self, generations: list[str]) -> dict:
        stats = defaultdict(lambda: {"n": 0, "correct": 0.0, "invalid": 0})
        
        for pred, gt in zip(generations, self.ground_truths):
            subject = gt["subject"]

            stats[subject]["n"] += 1
            pred_clean = pred.strip().upper()
            if pred_clean not in gt["valid_set"]:
                stats[subject]["invalid"] += 1
                stats[subject]["correct"] += 1 / len(gt["valid_set"])
            elif pred_clean == gt["gold"]:
                stats[subject]["correct"] += 1

        return compute_benchmark_micro_macro_avg(stats)

class HFMCQBenchmark(StandardMCQBenchmark):
    """Automates dataset loading and prompt formatting for standard HF datasets."""
    def __init__(
        self, path: str, split: str, shot_split: str, num_shots: int, 
        row_to_doc_fn: Callable[[dict[str, Any]], MCQDoc], name: Optional[str] = None
    ):
        super().__init__()
        # Load eval dataset
        eval_ds = load_dataset(path, name, split=split)
        eval_docs = [row_to_doc_fn(row) for row in eval_ds]  # pyright: ignore[reportArgumentType]
        
        # Load shot dataset efficiently using slice notation
        shot_docs = []
        if num_shots > 0:
            shot_ds = load_dataset(path, name, split=f"{shot_split}[:{num_shots}]")
            shot_docs = [row_to_doc_fn(row) for row in shot_ds]  # pyright: ignore[reportArgumentType]
            
        for doc in eval_docs:
            prompt, choices = _construct_mcq_prompt(doc, shot_docs)
            self.prompts.append(prompt)
            self.ground_truths.append({
                "valid_set": set(choices),
                "gold": chr(65 + doc.gold_index)
            })

class ARC(HFMCQBenchmark):
    def __init__(self, split: str = "test", subset: str = "ARC-Challenge", num_shots: int = 25):
        super().__init__(
            path="allenai/ai2_arc", name=subset, split=split, shot_split="validation", num_shots=num_shots,
            row_to_doc_fn=lambda r: MCQDoc(r['question'], r['choices']['text'], r['choices']['label'].index(r['answerKey']))
        )

class HellaSwag(HFMCQBenchmark):
    def __init__(self, split: str = "validation", num_shots: int = 10):
        super().__init__(
            path="Rowan/hellaswag", split=split, shot_split="train", num_shots=num_shots,
            row_to_doc_fn=lambda r: MCQDoc(
                f"{r['ctx_a']} {r['ctx_b'].capitalize()}\nQuestion: Which is the most logical continuation?", 
                r['endings'], int(r['label'])
            )
        )

class Winogrande(HFMCQBenchmark):
    def __init__(self, split: str = "validation", num_shots: int = 5):
        super().__init__(
            path="allenai/winogrande", name="winogrande_debiased", split=split, shot_split="train", num_shots=num_shots,
            row_to_doc_fn=lambda r: MCQDoc(r["sentence"], [r['option1'], r['option2']], int(r['answer']) - 1)
        )

class BoolQ(HFMCQBenchmark):
    def __init__(self, split: str = "validation", num_shots: int = 5):
        super().__init__(
            path="google/boolq", split=split, shot_split="train", num_shots=num_shots,
            row_to_doc_fn=lambda r: MCQDoc(
                f"{r['passage']}\nQuestion: {r['question'].capitalize()}?", 
                ["Yes", "No"], 1 - int(bool(r['answer']))
            )
        )

# --- Majority Voting ---

class AIMEMajorityVoting(BaseBenchmark):
    def __init__(self, year: str = "25", n: int = 64, pass_k: list[int] = [1, 10]):
        super().__init__()
        self.n = n
        self.pass_k = pass_k

        dataset = load_dataset(f"math-ai/aime{year}", split="test")
        for item in dataset:
            # Extract answer: 'answer' col -> Extract from 'solution' col
            gt: str = item.get("answer")  # type: ignore
            if gt is None and item.get("solution"):  # type: ignore
                gt = last_boxed_only_string(item["solution"])  # type: ignore
            
            for _ in range(self.n):
                self.prompts.append(item["problem"])  # type: ignore
            self.ground_truths.append(int(gt))

    def compute_metrics(self, generations: list[str]) -> dict:
        maj_pass_k, pass_k, total = {k: 0 for k in self.pass_k}, {k: 0.0 for k in self.pass_k}, 0
        result = {}
        
        for i, gt in enumerate(self.ground_truths):
            votes = []
            c = 0  # Total correct answers (c)
            for text in generations[i * self.n: (i + 1) * self.n]:
                boxed = last_boxed_only_string(text)
                if boxed is not None:
                    text = boxed

                c += verify(str(gt), parse(text))
                try:
                    ans = int(text)
                    assert 1 <= ans <= 999
                    votes.append(ans)
                except:
                    pass

            sorted_votes = Counter(votes).most_common()

            total += 1
            result[f"{i}_valid"] = len(votes)
            result[f"{i}_cons_count"] = sorted_votes[0][-1] if sorted_votes else 0
            
            for k in self.pass_k:
                # 1. Majority Voting (maj pass@k) - Evaluated on top-k most frequent valid answers
                if gt in [ans for ans, _ in sorted_votes[:k]]:
                    maj_pass_k[k] += 1
                
                # 2. Unbiased pass@k - Evaluated strictly against the raw generation count
                if self.n >= k:
                    prob_fail = 1.0
                    for j in range(k):
                        prob_fail *= (self.n - c - j) / (self.n - j)
                    pass_k[k] += 1.0 - prob_fail

        metrics = {f"maj_pass@{k}": v / max(1, total) for k, v in maj_pass_k.items()}
        metrics |= {f"pass@{k}": v / max(1, total) for k, v in pass_k.items()}
        return result | metrics


class HumanEval(BaseBenchmark):
    def __init__(self, split: str = "test"):
        super().__init__()

        dataset = load_dataset("evalplus/humanevalplus", split=split)
        for item in dataset:
            self.prompts.append(f"Complete the following Python function.\n```python\n{item['prompt']}\n```")
            self.ground_truths.append({
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "entry_point": item["entry_point"]
            })

    def _match_answer(self, task_data: dict, response: str) -> dict:
        def _function_exists(code: str, func_name: str) -> bool:
            try:
                tree = ast.parse(code)
                return any(isinstance(n, ast.FunctionDef) and n.name == func_name for n in ast.walk(tree))
            except SyntaxError:
                return False

        def _try_match(content: str, prefix: str, entrypoint: str) -> Optional[str]:
            blocks = [m[1] for m in re.findall(r"(`{3}.*?\n+)([\s\S]*?)(\n+`{3})", content)] + [content]
            for block in blocks:
                completion = prefix + block
                if _function_exists(completion, entrypoint):
                    return completion
            return None

        # 1. Match with truncated prefix
        include_prefix = task_data['prompt'].split('def')[0].strip() + "\n\n"
        result = _try_match(response, include_prefix, task_data["entry_point"])
        # 2. Match with full prompt prefix
        if not result:
            result = _try_match(response, task_data["prompt"], task_data["entry_point"])
        # 3. Fallback
        if not result:
            result = response

        return {"task_id": task_data["task_id"], "solution": result}

    def compute_metrics(self, generations: list[str]) -> dict:
        samples = [self._match_answer(gt, gen) for gt, gen in zip(self.ground_truths, generations)]

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = os.path.join(tmpdir, "samples.jsonl")
            with open(samples_path, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")

            # Run EvalPlus Docker container
            cmd = [
                "docker", "run", "--rm", 
                "-v", f"{tmpdir}:/app", 
                "ganler/evalplus:latest", 
                "evalplus.evaluate", 
                "--dataset", "humaneval", 
                "--samples", "/app/samples.jsonl"
            ]

            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"EvalPlus execution failed:\n{e.stderr}\n{e.stdout}")
                return {"error": "EvalPlus Docker execution failed. Make sure Docker daemon is running."}

            results_path = os.path.join(tmpdir, "samples_eval_results.json")
            with open(results_path, "r", encoding="utf-8") as f:
                eval_results = json.load(f)

            # Extract metrics from the "eval" block
            eval_data = eval_results.get("eval", {})
            total_tasks = len(eval_data)
            base_passed = sum(1 for attempts in eval_data.values() if attempts[0].get("base_status") == "pass")
            plus_passed = sum(1 for attempts in eval_data.values() if attempts[0].get("plus_status") == "pass")

            return {
                "n": total_tasks,
                "pass@1_base": base_passed / total_tasks,
                "pass@1_plus": plus_passed / total_tasks
            }
