import os
import re
import json
import torch
import httpx
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

class EmbeddingClient:
    def __init__(
        self,
        api_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_url = (api_url or os.getenv("EMBEDDING_API_URL", "")).rstrip("/")
        if not self.api_url:
            raise ValueError("Set EMBEDDING_API_URL to an OpenAI-compatible embeddings endpoint.")
        self.model = model or os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-8B")
        self._client = httpx.Client(timeout=timeout)

    def embed(self, text: str) -> List[float]:
        
        payload = {"input": text, "model": self.model}
        try:
            r = self._client.post(self.api_url, json=payload)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Embedding API request failed: {e}")
        data = r.json()
        try:
            return data["data"][0]["embedding"]
        except Exception as e:
            raise RuntimeError(f"Invalid embedding response: {data}") from e

class TcmRewardManager:
    def __init__(self, api_url: str | None = None, model: str | None = None):
        
        self.client = EmbeddingClient(api_url=api_url, model=model)
        self._embed_cache = {}

    @staticmethod
    def _normalize_pred_json_keys_for_gt(pred_json: Dict[str, Any], gt_json: Dict[str, Any]) -> Dict[str, Any]:
        
        p = dict(pred_json)

        
        if "辨证结果" in gt_json and "辨证结果" not in p and "分析结果" in p:
            p["辨证结果"] = p.pop("分析结果")
        
        if "分析结果" in gt_json and "分析结果" not in p and "辨证结果" in p:
            p["分析结果"] = p.pop("辨证结果")

        # If Pred contains both variants but GT contains only one, drop the extra to match key set.
        if "辨证结果" in gt_json and "分析结果" in p and "分析结果" not in gt_json:
            p.pop("分析结果", None)
        if "分析结果" in gt_json and "辨证结果" in p and "辨证结果" not in gt_json:
            p.pop("辨证结果", None)

        return p
        
    def extract_json(self, text):
        
        def strip_code_fences(s: str) -> str:
            # Remove a single markdown code-fence wrapper if present.
            fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
            return fenced.group(1) if fenced else s

        def find_first_json_object(s: str) -> str | None:
            # Find the first balanced {...} JSON object, handling nested braces and strings.
            start_positions = [m.start() for m in re.finditer(r"\{", s)]
            for start in start_positions:
                depth = 0
                in_string = False
                escape = False
                for i in range(start, len(s)):
                    ch = s[i]
                    if in_string:
                        if escape:
                            escape = False
                            continue
                        if ch == "\\":
                            escape = True
                            continue
                        if ch == '"':
                            in_string = False
                        continue

                    if ch == '"':
                        in_string = True
                        continue
                    if ch == "{":
                        depth += 1
                        continue
                    if ch == "}":
                        depth -= 1
                        if depth == 0:
                            return s[start : i + 1]
                        continue
                # If we get here, braces were unbalanced for this start.
            return None

        try:
            if not text:
                return None
            text = strip_code_fences(str(text).strip())
            candidate = find_first_json_object(text)
            if not candidate:
                return None
            return json.loads(candidate)
        except Exception:
            return None

    def get_semantic_similarity(self, str1, str2):
        
        if not str1 or not str2:
            return 0.0
        s1, s2 = str1.strip(), str2.strip()
        if s1 == s2:
            return 1.0
        
        def get_cached_embed(t):
            
            if len(self._embed_cache) > 5000:
                self._embed_cache.clear()
            if t not in self._embed_cache:
                self._embed_cache[t] = torch.tensor(self.client.embed(t)).to(torch.float32)
            return self._embed_cache[t]

        try:
            v1 = get_cached_embed(s1)
            v2 = get_cached_embed(s2)
        except Exception as e:
            print(f"Embedding error: {e}")
            return 0.0
        
        raw_sim = float(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item())

       
        if raw_sim >= 0.92:
            
            return 1.0
        elif raw_sim < 0.72:
           
            return raw_sim * 0.1
        else:
            
            return pow(raw_sim, 2.0)

    def score_json_objects(self, pred_json, gt_json, expected_keys=None):
        
        if not pred_json or not gt_json:
            return 0.0

        
        pred_json = self._normalize_pred_json_keys_for_gt(pred_json, gt_json)
        
        
        format_score = 1.0
        if expected_keys:
            pred_keys = set(pred_json.keys())
            target_keys = set(expected_keys)
            if pred_keys != target_keys:
                format_score -= 0.3 

        
        for length_key, limit in [("简短原因", 50), ("综合结论", 40)]:
            if length_key in pred_json and len(str(pred_json[length_key])) > limit:
                format_score -= 0.2

        
        compare_keys = [k for k in gt_json.keys() if k != "简短原因"]

        scores = []
        for key in compare_keys:
            if key in pred_json:
                val_pred = str(pred_json[key])
                val_gt = str(gt_json[key])
                scores.append(self.get_semantic_similarity(val_pred, val_gt))
            else:
                scores.append(0.0)

        base_score = sum(scores) / len(scores) if scores else 0.0
        return max(0.0, base_score * format_score)

    def __call__(self, solution_str, ground_truth):
        
        reward = 0.0
        
        
        format_pattern = r"<reasoning>.*?<analysis>.*?</reasoning>"
        if not re.search(format_pattern, solution_str, re.DOTALL):
            return 0.0 
        
        reward += 0.1 

       
        action_regex = r"^action(\d+)\s*:\s*(?:\[(.*?)\]\s*:\s*)?(.*)$"

        pred_actions = re.findall(action_regex, solution_str, re.MULTILINE)
        gt_actions = re.findall(action_regex, ground_truth, re.MULTILINE)
        
        
        # a is (idx, optional_label, content)
        pred_action_dict = {a[0]: a[2] for a in pred_actions}
        gt_action_dict = {a[0]: a[2] for a in gt_actions}

        action_scores = []
        for idx, gt_content in gt_action_dict.items():
            if idx == "1": continue 
            
            if idx in pred_action_dict:
                pred_content = pred_action_dict[idx]
                
                p_json = self.extract_json(pred_content)
                g_json = self.extract_json(gt_content)
                
                if p_json and g_json:
                    
                    expected = list(g_json.keys())
                    a_score = self.score_json_objects(p_json, g_json, expected_keys=expected)
                    
                    
                    gt_res = g_json.get("辨证结果") or g_json.get("分析结果")
                    pred_res = p_json.get("辨证结果") or p_json.get("分析结果")
                    if isinstance(gt_res, str) and "/" in gt_res:
                        if isinstance(pred_res, str) and "/" in pred_res:
                            
                            if len(pred_res.split("/")) == len(gt_res.split("/")):
                                a_score = min(1.0, a_score + 0.1)
                    
                    action_scores.append(a_score)
                else:
                    action_scores.append(self.get_semantic_similarity(pred_content, gt_content))
            else:
                action_scores.append(0.0)
        
        if action_scores:
            reward += (sum(action_scores) / len(action_scores)) * 0.5 

        
        try:
            
            pred_summary_part = solution_str.split("</reasoning>")[-1]
            gt_summary_part = ground_truth.split("</reasoning>")[-1]
            
            p_sum_json = self.extract_json(pred_summary_part)
            g_sum_json = self.extract_json(gt_summary_part)
            
            if p_sum_json and g_sum_json:
                
                summary_score = self.score_json_objects(p_sum_json, g_sum_json, expected_keys=list(g_sum_json.keys()))
                reward += summary_score * 0.4 
            else:
                reward += self.get_semantic_similarity(pred_summary_part, gt_summary_part) * 0.2
        except:
            pass

        return reward


tcm_reward_manager = TcmRewardManager()

REWARD_TYPE = "sequential"
REWARD_NAME = "tcm_bagang_reward"

def compute_score(data: Dict[str, Any], **kwargs) -> Dict[str, float]:

    responses = data.get("responses", data.get("response", ""))
    solution_str = responses[0] if isinstance(responses, list) else responses
    

    ground_truth = data.get("ground_truth") or data.get("output") or ""
    
    score = tcm_reward_manager(solution_str, ground_truth)
    return {"overall": score, "accuracy": score}
