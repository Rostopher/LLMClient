#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervisor 测试用模拟脚本

模拟 LLMClient 的 stdout 输出模式，用于验证 supervisor.py 的熔断逻辑。

用法:
    python _test_supervisor_mock.py [scenario]

场景:
    success     - 全部成功，正常退出
    gradual     - 逐渐增多的失败，最终触发熔断
    burst       - 突发连续失败，触发连续失败上限
    mixed       - 混合成功失败，不触发熔断
"""

import sys
import time
import uuid


def make_call_id():
    return f"call_{uuid.uuid4().hex[:16]}"


def print_success(call_id, ms=150):
    print(f"[{call_id}] 调用API: test_api, 模型: test-model, 重试: 0/3")
    print(f"[{call_id}] ✅ 调用成功 ({ms}ms)")
    print(f"[{call_id}] Token: 100+50=150")


def print_failure_api(call_id, msg="Request timed out."):
    print(f"[{call_id}] 调用API: test_api, 模型: test-model, 重试: 0/3")
    print(f"[{call_id}] ⚠️ API连接失败: {msg}")
    print(f"[{call_id}] 等待 1 秒后重试...")
    print(f"[{call_id}] 调用API: test_api, 模型: test-model, 重试: 1/3")
    print(f"[{call_id}] ❌ 调用失败: 连接错误: {msg}")


def print_batch_success(i, total, uid):
    print(f"[{i}/{total}] ✅ 成功: {uid} (call_id: {make_call_id()})")


def print_batch_failure(i, total, uid, msg="timeout"):
    print(f"[{i}/{total}] ❌ 失败: {uid} - {msg}")


def scenario_success():
    print("🧮 待处理任务: 20，批次数: 2")
    for i in range(1, 21):
        cid = make_call_id()
        print_success(cid, ms=100 + i * 10)
        time.sleep(0.1)
    print("✅ 批次 1/1 完成")
    print("============================================================")
    print("批量处理完成")
    print("============================================================")


def scenario_gradual():
    """先成功5个，然后逐渐失败，最终连续失败触发熔断"""
    print("🧮 待处理任务: 100，批次数: 5")
    for i in range(5):
        print_success(make_call_id())
        time.sleep(0.05)

    for i in range(30):
        if i % 5 == 0:
            print_success(make_call_id())
        else:
            print_failure_api(make_call_id())
        time.sleep(0.1)

    # 此时应该已经触发熔断，如果没有，继续全部失败
    for i in range(20):
        print_failure_api(make_call_id())
        time.sleep(0.05)


def scenario_burst():
    """5个成功后突发20个连续失败"""
    print("🧮 待处理任务: 50，批次数: 3")
    for i in range(5):
        print_batch_success(i + 1, 50, f"task_{i:03d}")
        time.sleep(0.05)

    for i in range(20):
        print_batch_failure(i + 6, 50, f"task_{i+5:03d}", msg="API rate limit exceeded")
        time.sleep(0.05)


def scenario_mixed():
    """混合成功失败，比例不触发熔断"""
    print("🧮 待处理任务: 30，批次数: 2")
    for i in range(30):
        if i % 4 == 0:
            print_failure_api(make_call_id())
        else:
            print_success(make_call_id())
        time.sleep(0.1)
    print("✅ 批次 1/1 完成")


SCENARIOS = {
    "success": scenario_success,
    "gradual": scenario_gradual,
    "burst": scenario_burst,
    "mixed": scenario_mixed,
}

if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "gradual"
    if scenario not in SCENARIOS:
        print(f"未知场景: {scenario}，可选: {', '.join(SCENARIOS)}")
        sys.exit(1)

    print(f"[MOCK] 模拟场景: {scenario}", flush=True)
    sys.stdout.flush()
    SCENARIOS[scenario]()
    sys.stdout.flush()
    print(f"[MOCK] 场景 {scenario} 结束")

    if scenario == "success" or scenario == "mixed":
        sys.exit(0)
    else:
        sys.exit(1)
