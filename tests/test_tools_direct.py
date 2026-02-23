"""Direct test of all 11 built-in tools.
Tests execute_builtin_tool directly without AI model dependency.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from builtin_tools import execute_builtin_tool, is_builtin_tool, BUILTIN_TOOL_NAMES

TEST_SESSION = "test_direct_session"
CONTEXT = {
    "api_key": "test_direct_key",
    "session_id": TEST_SESSION,
    "username": "admin"
}


def parse_result(result_str):
    try:
        return json.loads(result_str)
    except:
        return {"raw": result_str}


def test_tool(name, args, check_fn=None):
    print(f"\n--- {name} ---")
    assert is_builtin_tool(name), f"{name} not recognized as builtin tool"
    result_str = execute_builtin_tool(name, args, CONTEXT)
    result = parse_result(result_str)
    print(f"  Result: {json.dumps(result, ensure_ascii=False)[:300]}")

    if "error" in result and not (check_fn and check_fn(result)):
        print(f"  STATUS: FAIL (error: {result['error']})")
        return False

    if check_fn:
        passed = check_fn(result)
        print(f"  STATUS: {'PASS' if passed else 'FAIL'}")
        return passed

    print(f"  STATUS: PASS")
    return True


def main():
    print("=" * 60)
    print("  DIRECT TOOL EXECUTION TEST (no AI model needed)")
    print("=" * 60)

    results = {}

    results["run_code"] = test_tool("run_code", {"code": "print(sum(range(1, 11)))"}, 
        lambda r: r.get("status") == "success" and "55" in r.get("stdout", ""))

    results["web_search"] = test_tool("web_search", {"query": "Python programming", "max_results": 3},
        lambda r: isinstance(r.get("results"), list) and len(r.get("results", [])) > 0)

    results["http_request"] = test_tool("http_request", {"url": "https://httpbin.org/get", "method": "GET"},
        lambda r: r.get("status_code") == 200)

    results["debug_code"] = test_tool("debug_code", {"code": "x = 1/0\nprint(x)", "language": "python"},
        lambda r: "error" in str(r).lower() or "ZeroDivisionError" in str(r))

    results["file_write"] = test_tool("file_write", {"filename": "direct_test.txt", "content": "Direct test content 12345"},
        lambda r: r.get("status") == "success")

    results["file_read"] = test_tool("file_read", {"filename": "direct_test.txt"},
        lambda r: r.get("status") == "success" and "12345" in r.get("content", ""))

    results["apply_patch"] = test_tool("apply_patch", {
        "filename": "direct_test.txt",
        "patches": [{"operation": "replace", "find": "12345", "content": "PATCHED"}]
    }, lambda r: r.get("status") == "success" and r.get("applied_count", 0) > 0)

    file_read_after = parse_result(execute_builtin_tool("file_read", {"filename": "direct_test.txt"}, CONTEXT))
    patch_verified = "PATCHED" in file_read_after.get("content", "")
    print(f"  Patch verify: {'PATCHED found' if patch_verified else 'PATCHED not found'}")

    results["list_directory"] = test_tool("list_directory", {"path": "."},
        lambda r: r.get("status") == "success")

    results["create_directory"] = test_tool("create_directory", {"path": "test_subdir_direct"},
        lambda r: r.get("status") == "success")

    results["run_shell"] = test_tool("run_shell", {"command": 'echo "hello shell test"'},
        lambda r: r.get("status") == "success" and "hello" in r.get("stdout", ""))

    results["install_package"] = test_tool("install_package", {"package": "requests"},
        lambda r: r.get("status") == "success" or "already satisfied" in r.get("stdout", "").lower())

    results["memory_write"] = test_tool("memory_write", {"key": "test_mem", "value": "memory value 999"},
        lambda r: r.get("status") == "success")

    results["memory_read"] = test_tool("memory_read", {"key": "test_mem"},
        lambda r: r.get("status") == "success" and "999" in r.get("value", ""))

    results["database_query"] = test_tool("database_query", {"operation": "set", "key": "test_db", "value": "db_val_777"},
        lambda r: r.get("status") == "success")

    results["task_status"] = test_tool("task_status", {"detail_level": "summary"},
        lambda r: True)

    print("\n" + "=" * 60)
    print("  DIRECT TEST RESULTS")
    print("=" * 60)
    passed = 0
    core_tools = ["run_code", "web_search", "http_request", "debug_code", 
                  "file_write", "file_read", "apply_patch", "list_directory",
                  "create_directory", "run_shell", "install_package"]
    
    for name in core_tools:
        status = "PASS" if results.get(name) else "FAIL"
        print(f"  {name:25s} : {status}")
        if results.get(name):
            passed += 1

    print(f"\n  CORE TOOLS: {passed}/{len(core_tools)} PASS")
    
    extra_passed = sum(1 for k in ["memory_write", "memory_read", "database_query", "task_status"] if results.get(k))
    print(f"  EXTRA TOOLS: {extra_passed}/4 PASS")
    print("=" * 60)

    return passed == len(core_tools)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
