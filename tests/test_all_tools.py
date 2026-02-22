"""Real integration test for all 11 built-in tools via the Agent API.
Tests actual tool execution, not simulation.
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:5000"
API_KEY = "sk-dzeck-3a1bcd62bb43ec3f2458ea8a149d22296b78f68ab9583c66"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def call_agent(user_message, tool_choice="auto", stream=False, extra_tools=None):
    payload = {
        "messages": [{"role": "user", "content": user_message}],
        "tool_choice": tool_choice,
        "stream": stream,
        "builtin_tools": True,
        "enable_planning": False,
        "enable_reflection": False,
        "max_iterations": 5
    }
    if extra_tools:
        payload["tools"] = extra_tools

    resp = requests.post(f"{BASE_URL}/v1/agent/completions", headers=HEADERS, json=payload, timeout=120)
    return resp.json()


def check_tool_called(result, tool_name):
    agent_loop = result.get("agent_loop", {})
    tool_calls = agent_loop.get("tool_calls", [])
    for tc in tool_calls:
        if tc.get("name") == tool_name:
            return True, tc.get("result_preview", "")
    return False, ""


def test_run_code():
    print("\n=== TEST 1: run_code (execute real Python code) ===")
    result = call_agent("Execute this Python code and show the output: print([x**2 for x in range(1, 8)])")
    called, preview = check_tool_called(result, "run_code")
    print(f"TOOL: run_code -> {'CALLED' if called else 'NOT CALLED'}")
    if called:
        print(f"OUTPUT: {preview[:200]}")
    print(f"STATUS: {'REAL EXECUTION' if called else 'FAILED - NO TOOL CALL'}")
    return called


def test_web_search():
    print("\n=== TEST 2: web_search (real web search) ===")
    result = call_agent('Use the web_search tool to search for "Python Flask framework"')
    called, preview = check_tool_called(result, "web_search")
    print(f"TOOL: web_search -> {'CALLED' if called else 'NOT CALLED'}")
    if called:
        print(f"RESULT: {preview[:200]}")
    print(f"STATUS: {'REAL SEARCH' if called else 'FAILED - NO TOOL CALL'}")
    return called


def test_debug_code():
    print("\n=== TEST 3: debug_code (real code analysis) ===")
    buggy_code = """def calculate(x, y):
    result = x / y
    return result

print(calculate(10, 0))"""
    result = call_agent(f'Use the debug_code tool to analyze this Python code for bugs:\n```python\n{buggy_code}\n```')
    called, preview = check_tool_called(result, "debug_code")
    print(f"TOOL: debug_code -> {'CALLED' if called else 'NOT CALLED'}")
    if called:
        print(f"ANALYSIS: {preview[:300]}")
    print(f"STATUS: {'REAL ANALYSIS' if called else 'FAILED - NO TOOL CALL'}")
    return called


def test_apply_patch():
    print("\n=== TEST 4: apply_patch (real file patching) ===")
    call_agent('Use the file_write tool to write a file called "test_patch.py" with this content: def hello():\n    print("Hello World")')
    time.sleep(2)
    result = call_agent('Use the apply_patch tool to patch the file "test_patch.py": replace "Hello World" with "Hello Dzeck AI"')
    called, preview = check_tool_called(result, "apply_patch")
    print(f"TOOL: apply_patch -> {'CALLED' if called else 'NOT CALLED'}")
    if called:
        print(f"PATCH RESULT: {preview[:200]}")
    print(f"STATUS: {'REAL PATCH' if called else 'FAILED - NO TOOL CALL'}")
    return called


def test_http_request():
    print("\n=== TEST 5: http_request (real HTTP call) ===")
    result = call_agent('Use the http_request tool to make a GET request to https://httpbin.org/get')
    called, preview = check_tool_called(result, "http_request")
    print(f"TOOL: http_request -> {'CALLED' if called else 'NOT CALLED'}")
    if called:
        print(f"RESPONSE: {preview[:200]}")
    print(f"STATUS: {'REAL HTTP' if called else 'FAILED - NO TOOL CALL'}")
    return called


def test_memory_write_read():
    print("\n=== TEST 6: memory_write + memory_read (real memory ops) ===")
    result = call_agent('Use the memory_write tool to store key "test_key" with value "Dzeck AI is working"')
    called_w, preview_w = check_tool_called(result, "memory_write")
    print(f"TOOL: memory_write -> {'CALLED' if called_w else 'NOT CALLED'}")

    time.sleep(1)
    result2 = call_agent('Use the memory_read tool to read the key "test_key"')
    called_r, preview_r = check_tool_called(result2, "memory_read")
    print(f"TOOL: memory_read -> {'CALLED' if called_r else 'NOT CALLED'}")
    if called_r:
        print(f"MEMORY VALUE: {preview_r[:200]}")
    print(f"STATUS: {'REAL MEMORY OPS' if called_w and called_r else 'PARTIAL' if called_w or called_r else 'FAILED'}")
    return called_w and called_r


def test_file_write_read():
    print("\n=== TEST 7: file_write + file_read (real file ops) ===")
    result = call_agent('Use the file_write tool to create a file named "test_file.txt" with content "Api Dzeck Ai test file - tool execution verified"')
    called_w, preview_w = check_tool_called(result, "file_write")
    print(f"TOOL: file_write -> {'CALLED' if called_w else 'NOT CALLED'}")

    time.sleep(1)
    result2 = call_agent('Use the file_read tool to read the file "test_file.txt"')
    called_r, preview_r = check_tool_called(result2, "file_read")
    print(f"TOOL: file_read -> {'CALLED' if called_r else 'NOT CALLED'}")
    if called_r:
        print(f"FILE CONTENT: {preview_r[:200]}")
    print(f"STATUS: {'REAL FILE OPS' if called_w and called_r else 'PARTIAL' if called_w or called_r else 'FAILED'}")
    return called_w and called_r


def test_database_query():
    print("\n=== TEST 8: database_query (real DB ops) ===")
    result = call_agent('Use the database_query tool with operation "set", key "test_db_key", value "database test value 123"')
    called, preview = check_tool_called(result, "database_query")
    print(f"TOOL: database_query -> {'CALLED' if called else 'NOT CALLED'}")
    if called:
        print(f"DB RESULT: {preview[:200]}")
    print(f"STATUS: {'REAL DB' if called else 'FAILED - NO TOOL CALL'}")
    return called


def test_task_status():
    print("\n=== TEST 9: task_status (real status check) ===")
    result = call_agent('Use the task_status tool with detail_level "summary" to check current task status')
    called, preview = check_tool_called(result, "task_status")
    print(f"TOOL: task_status -> {'CALLED' if called else 'NOT CALLED'}")
    if called:
        print(f"STATUS: {preview[:200]}")
    print(f"STATUS: {'REAL STATUS' if called else 'FAILED - NO TOOL CALL'}")
    return called


def main():
    print("=" * 60)
    print("  Api Dzeck Ai - REAL TOOL EXECUTION TEST")
    print(f"  API Key: {API_KEY[:20]}...")
    print(f"  Base URL: {BASE_URL}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\nChecking server health...")
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=10)
        h = health.json()
        print(f"Server: OK | Providers: {h.get('available_providers')} | Models: {h.get('total_models')}")
    except Exception as e:
        print(f"Server not available: {e}")
        sys.exit(1)

    results = {}
    tests = [
        ("run_code", test_run_code),
        ("web_search", test_web_search),
        ("debug_code", test_debug_code),
        ("apply_patch", test_apply_patch),
        ("http_request", test_http_request),
        ("memory_write+read", test_memory_write_read),
        ("file_write+read", test_file_write_read),
        ("database_query", test_database_query),
        ("task_status", test_task_status),
    ]

    for name, test_fn in tests:
        try:
            time.sleep(3)
            results[name] = test_fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    passed = 0
    total = len(results)
    for name, success in results.items():
        status = "PASS (REAL)" if success else "FAIL"
        print(f"  {name:25s} : {status}")
        if success:
            passed += 1

    print(f"\n  TOTAL: {passed}/{total} tools working with REAL execution")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
