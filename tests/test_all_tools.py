"""Real integration test for all 11 built-in tools via /v1/agent/completions.
Tests actual tool execution, not simulation.
Uses X-Admin-Key header for auto-authentication.
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:5000"
ADMIN_KEY = "dzeckaiv1"

HEADERS = {
    "X-Admin-Key": ADMIN_KEY,
    "Content-Type": "application/json"
}

def call_agent(user_message, tool_choice="auto", stream=False):
    payload = {
        "messages": [{"role": "user", "content": user_message}],
        "tool_choice": tool_choice,
        "stream": stream,
        "builtin_tools": True,
        "enable_planning": False,
        "enable_reflection": False,
        "max_iterations": 5
    }

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
    print("\n=== TEST 1: run_code ===")
    result = call_agent('Use the run_code tool to execute this Python code: print([x**2 for x in range(1, 8)])')
    called, preview = check_tool_called(result, "run_code")
    print(f"  run_code -> {'PASS' if called else 'FAIL'}")
    if called:
        print(f"  Output: {preview[:200]}")
    return called


def test_web_search():
    print("\n=== TEST 2: web_search ===")
    result = call_agent('Use the web_search tool to search for "Python Flask framework"')
    called, preview = check_tool_called(result, "web_search")
    print(f"  web_search -> {'PASS' if called else 'FAIL'}")
    if called:
        print(f"  Output: {preview[:200]}")
    return called


def test_debug_code():
    print("\n=== TEST 3: debug_code ===")
    buggy_code = 'def calc(x, y):\n    return x / y\nprint(calc(10, 0))'
    result = call_agent(f'Use the debug_code tool to debug this Python code:\n```python\n{buggy_code}\n```')
    called, preview = check_tool_called(result, "debug_code")
    print(f"  debug_code -> {'PASS' if called else 'FAIL'}")
    if called:
        print(f"  Output: {preview[:200]}")
    return called


def test_http_request():
    print("\n=== TEST 4: http_request ===")
    result = call_agent('Use the http_request tool to make a GET request to https://httpbin.org/get')
    called, preview = check_tool_called(result, "http_request")
    print(f"  http_request -> {'PASS' if called else 'FAIL'}")
    if called:
        print(f"  Output: {preview[:200]}")
    return called


def test_file_write():
    print("\n=== TEST 5: file_write (write_file) ===")
    result = call_agent('Use the file_write tool to create a file named "test_hello.txt" with content "Hello from Dzeck AI tools test"')
    called, preview = check_tool_called(result, "file_write")
    print(f"  file_write -> {'PASS' if called else 'FAIL'}")
    if called:
        print(f"  Output: {preview[:200]}")
    return called


def test_file_read():
    print("\n=== TEST 6: file_read (read_file) ===")
    result = call_agent('Use the file_read tool to read the file "test_hello.txt"')
    called, preview = check_tool_called(result, "file_read")
    print(f"  file_read -> {'PASS' if called else 'FAIL'}")
    if called:
        print(f"  Output: {preview[:200]}")
    return called


def test_apply_patch():
    print("\n=== TEST 7: apply_patch ===")
    call_agent('Use the file_write tool to write a file called "patch_test.py" with content "def greet():\n    print(\'Hello World\')"')
    time.sleep(2)
    result = call_agent('Use the apply_patch tool on file "patch_test.py" with operation "replace", find "Hello World", content "Hello Dzeck AI"')
    called, preview = check_tool_called(result, "apply_patch")
    print(f"  apply_patch -> {'PASS' if called else 'FAIL'}")
    if called:
        print(f"  Output: {preview[:200]}")
    return called


def test_list_directory():
    print("\n=== TEST 8: list_directory ===")
    result = call_agent('Use the list_directory tool to list all files in the current directory (path ".")')
    called, preview = check_tool_called(result, "list_directory")
    print(f"  list_directory -> {'PASS' if called else 'FAIL'}")
    if called:
        print(f"  Output: {preview[:200]}")
    return called


def test_create_directory():
    print("\n=== TEST 9: create_directory ===")
    result = call_agent('Use the create_directory tool to create a directory named "test_folder_abc"')
    called, preview = check_tool_called(result, "create_directory")
    print(f"  create_directory -> {'PASS' if called else 'FAIL'}")
    if called:
        print(f"  Output: {preview[:200]}")
    return called


def test_run_shell():
    print("\n=== TEST 10: run_shell ===")
    result = call_agent('Use the run_shell tool to execute the command: echo "Hello from shell test"')
    called, preview = check_tool_called(result, "run_shell")
    print(f"  run_shell -> {'PASS' if called else 'FAIL'}")
    if called:
        print(f"  Output: {preview[:200]}")
    return called


def test_install_package():
    print("\n=== TEST 11: install_package ===")
    result = call_agent('Use the install_package tool to install the Python package "requests"')
    called, preview = check_tool_called(result, "install_package")
    print(f"  install_package -> {'PASS' if called else 'FAIL'}")
    if called:
        print(f"  Output: {preview[:200]}")
    return called


def main():
    print("=" * 60)
    print("  Api Dzeck Ai - ALL 11 TOOLS TEST")
    print(f"  Endpoint: /v1/agent/completions")
    print(f"  Auth: X-Admin-Key header")
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
        ("http_request", test_http_request),
        ("file_write", test_file_write),
        ("file_read", test_file_read),
        ("apply_patch", test_apply_patch),
        ("list_directory", test_list_directory),
        ("create_directory", test_create_directory),
        ("run_shell", test_run_shell),
        ("install_package", test_install_package),
    ]

    for name, test_fn in tests:
        try:
            time.sleep(3)
            results[name] = test_fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    passed = 0
    total = len(results)
    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {name:25s} : {status}")
        if success:
            passed += 1

    print(f"\n  TOTAL: {passed}/{total} tools PASS")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
