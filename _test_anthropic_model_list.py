from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

# 确保从仓库根目录导入（避免以脚本方式运行时找不到包）
current_dir = Path(__file__).parent.resolve()
repo_root = current_dir.parent.resolve()
sys.path.insert(0, str(repo_root))

from LLMClient_v2.anthropic_client import AnthropicLLMClient  # noqa: E402
from LLMClient_v2.llm_runtime_config import list_runtime_profiles  # noqa: E402


def build_prompt(model_name: str) -> str:
    return (
        "请严格按两行输出：\n"
        "1) 第一行：MODEL_SELF_REPORT=<你认为你正在运行的模型名；不知道就写unknown>\n"
        "2) 第二行：OK\n"
        f"现在我请求的模型是：{model_name}\n"
    )


async def test_models(api_name: str, models: List[str]) -> List[Dict[str, Any]]:
    client = AnthropicLLMClient(api_name=api_name)
    # 测试脚本：避免失败时指数退避等太久
    client.max_retries = 1
    client.retry_base_delay = 0.5
    results: List[Dict[str, Any]] = []

    for model_name in models:
        resp = await client.get_completion(
            prompt=build_prompt(model_name),
            model=model_name,
            max_tokens=64,
            temperature=0.0,
            metadata={"test": "anthropic_model_list"},
        )

        results.append(
            {
                "api_name": api_name,
                "requested_model": model_name,
                "success": resp.success,
                "raw_model": resp.raw_model,
                "usage": asdict(resp.usage) if resp.usage else None,
                "cost": resp.cost,
                "error": resp.error,
                "content_head": (resp.content or "").splitlines()[:5],
            }
        )

    return results


def main() -> None:
    api_name = "anthropic_antigravity_local"
    profiles = list_runtime_profiles()
    config = profiles.get(api_name)
    if not config:
        raise ValueError(f"{api_name} 未在 llm_apis.yaml 中配置")
    model_list = config.get("model_list")
    if not isinstance(model_list, list) or not model_list:
        raise ValueError(f"{api_name} 未配置有效的 model_list")

    results = asyncio.run(test_models(api_name, model_list))

    ok = [r for r in results if r["success"]]
    bad = [r for r in results if not r["success"]]

    print(f"API={api_name} | 总模型数={len(results)} | 成功={len(ok)} | 失败={len(bad)}")
    print("前 10 条结果（含 raw_model 与 self-report 首行）：")
    for r in results[:10]:
        first_line = r["content_head"][0] if r["content_head"] else ""
        print(f"- req={r['requested_model']} | raw_model={r['raw_model']} | {first_line}")

    output_path = Path(__file__).parent.resolve() / "anthropic_model_list_test_results.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详细结果已写入：{output_path}")


if __name__ == "__main__":
    main()
