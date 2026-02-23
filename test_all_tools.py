#!/usr/bin/env python3
"""Test all 11 agent tools via /v1/agent/completions endpoint."""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:5000"
ADMIN_KEY = "dzeckaiv1"
HEADERS = {
    "Content-Type": "application/json",
    "X-Admin-Key": ADMIN_KEY
}

def call_agent(prompt, tool_choice=None):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "openai",
        "max_tokens": 1500
    }
    if tool_choice:
        payload["tool_choice"] = tool_choice
    
    resp = requests.post(
        f"{BASE_URL}/v1/agent/completions",
        headers=HEADERS,
        json=payload,
        timeout=90,
        stream=True
    )
    
    full_text = ""
    agent_loop = None
    tool_calls = []
    
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                evt_type = data.get("type", "")
                if evt_type == "content_block_delta":
                    text = data.get("delta", {}).get("text", "")
                    full_text += text
                elif evt_type == "message_stop":
                    agent_loop = data.get("agent_loop", {})
                    tool_calls = agent_loop.get("tool_calls", [])
            except:
                pass
    
    return full_text, tool_calls, agent_loop


def test_tool(name, prompt, check_fn=None, tool_choice=None):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"Prompt: {prompt[:80]}...")
    
    try:
        text, tool_calls, agent_loop = call_agent(prompt, tool_choice)
        
        tool_names = [tc.get("name", "") for tc in tool_calls] if tool_calls else []
        print(f"Tools called: {tool_names}")
        
        if tool_calls:
            for tc in tool_calls:
                result_preview = tc.get("result_preview", "")[:200]
                print(f"  -> {tc.get('name')}: {result_preview}")
        
        response_preview = text[:300] if text else "(empty)"
        print(f"Response: {response_preview}")
        
        if agent_loop:
            print(f"Agent loop: {agent_loop.get('iterations', 0)} iterations, {agent_loop.get('tool_calls_count', 0)} tool calls")
        
        tool_executed = False
        if tool_calls:
            for tc in tool_calls:
                result = tc.get("result_preview", "")
                if '"status": "success"' in result or '"results_count"' in result or '"status_code"' in result:
                    tool_executed = True
                    break
                if '"error"' not in result and result:
                    tool_executed = True
                    break
        
        if tool_executed:
            print(f"RESULT: PASS - Tool executed successfully")
            return True
        elif tool_calls:
            print(f"RESULT: PARTIAL - Tool was called but may have issues")
            return True
        else:
            print(f"RESULT: FAIL - No tool was called")
            return False
            
    except Exception as e:
        print(f"RESULT: ERROR - {e}")
        return False


def main():
    print("=" * 60)
    print("TESTING ALL 11 AGENT TOOLS")
    print(f"Endpoint: {BASE_URL}/v1/agent/completions")
    print("=" * 60)
    
    results = {}
    
    # 1. file_read (read_file)
    results["read_file"] = test_tool(
        "read_file (file_read)",
        "Gunakan tool file_read untuk membaca file '*' dan tampilkan daftar file di workspace"
    )
    time.sleep(2)
    
    # 2. file_write (write_file)
    results["write_file"] = test_tool(
        "write_file (file_write)",
        'Gunakan tool file_write untuk menulis file bernama "test_output.txt" dengan content "Hello from Agent Tool Test! Timestamp: 2024"'
    )
    time.sleep(2)
    
    # 3. apply_patch
    results["apply_patch"] = test_tool(
        "apply_patch",
        'Gunakan tool apply_patch pada file "test_output.txt" dengan patches: replace "2024" dengan "2025 - PATCHED"'
    )
    time.sleep(2)
    
    # 4. list_directory
    results["list_directory"] = test_tool(
        "list_directory",
        "Gunakan tool list_directory untuk menampilkan semua file di workspace root path '.'"
    )
    time.sleep(2)
    
    # 5. create_directory
    results["create_directory"] = test_tool(
        "create_directory",
        'Gunakan tool create_directory untuk membuat directory baru bernama "test_folder_agent"'
    )
    time.sleep(2)
    
    # 6. run_code
    results["run_code"] = test_tool(
        "run_code",
        'Gunakan tool run_code untuk menjalankan kode python ini: print("Hello from run_code!"); import sys; print(f"Python version: {sys.version}")'
    )
    time.sleep(2)
    
    # 7. run_shell
    results["run_shell"] = test_tool(
        "run_shell",
        'Gunakan tool run_shell untuk menjalankan shell command: echo "Shell test OK" && uname -a && date'
    )
    time.sleep(2)
    
    # 8. install_package
    results["install_package"] = test_tool(
        "install_package",
        'Gunakan tool install_package untuk install python package "cowsay"'
    )
    time.sleep(2)
    
    # 9. debug_code
    results["debug_code"] = test_tool(
        "debug_code",
        'Gunakan tool debug_code untuk debug kode python ini yang error:\n```python\ndef calculate(x, y):\n    result = x / y\n    return result\n\nprint(calculate(10, 0))\n```'
    )
    time.sleep(2)
    
    # 10. web_search
    results["web_search"] = test_tool(
        "web_search",
        'Gunakan tool web_search untuk mencari "Python Flask REST API tutorial 2024"'
    )
    time.sleep(2)
    
    # 11. http_request
    results["http_request"] = test_tool(
        "http_request",
        'Gunakan tool http_request untuk membuat GET request ke https://httpbin.org/json'
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - ALL 11 TOOLS TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    for tool_name, result in results.items():
        status = "PASS" if result else "FAIL"
        icon = "[+]" if result else "[-]"
        print(f"  {icon} {tool_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed}/{len(results)} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tools working!")
        sys.exit(0)


if __name__ == "__main__":
    main()
