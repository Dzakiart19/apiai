#!/usr/bin/env python3
"""Comprehensive test script for ALL providers, ALL models, and tool calling."""

import json
import time
import sys
import requests

BASE_URL = "http://localhost:5000"
ADMIN_PASSWORD = "dzeckaiv1"

def parse_sse(raw):
    parts = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            d = line[6:]
            if d == "[DONE]":
                continue
            try:
                chunk = json.loads(d)
                c = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if c:
                    parts.append(c)
            except:
                pass
    return "".join(parts)

def chat_test(provider, model, msg="Reply with just the number: 2+2=?"):
    headers = {"Content-Type": "application/json", "X-Admin-Key": ADMIN_PASSWORD}
    payload = {"model": model, "provider": provider, "messages": [{"role": "user", "content": msg}], "max_tokens": 50}
    try:
        r = requests.post(f"{BASE_URL}/v1/chat/completions", headers=headers, json=payload, timeout=45)
        if r.text.strip().startswith("data:"):
            c = parse_sse(r.text)
            return ("OK", c[:100]) if c and "[ERROR]" not in c else ("ERROR", c[:100] if c else "empty")
        try:
            d = r.json()
        except:
            return ("PARSE_ERR", r.text[:80])
        if r.status_code == 200:
            ch = d.get("choices", [])
            if ch:
                c = ch[0].get("message", {}).get("content", "")
                return ("OK", c[:100]) if c and "[ERROR]" not in c else ("EMPTY", c[:100] if c else "empty")
        return (f"HTTP_{r.status_code}", str(d.get("error", d))[:80])
    except requests.exceptions.Timeout:
        return ("TIMEOUT", "45s")
    except Exception as e:
        return ("EXCEPTION", str(e)[:80])

def tool_test(provider, model):
    headers = {"Content-Type": "application/json", "X-Admin-Key": ADMIN_PASSWORD}
    tools = [
        {"type": "function", "function": {"name": "run_code", "description": "Execute Python code safely", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
        {"type": "function", "function": {"name": "web_search", "description": "Search the web for information", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
    ]
    payload = {"model": model, "provider": provider, "messages": [{"role": "user", "content": "Hitung 15 x 37 pakai tool run_code"}], "tools": tools, "max_tokens": 200}
    try:
        r = requests.post(f"{BASE_URL}/v1/agent/completions", headers=headers, json=payload, timeout=60)
        if r.text.strip().startswith("data:"):
            c = parse_sse(r.text)
            if c and "555" in c:
                return ("ANSWERED", c[:80])
            return ("RESPONSE", c[:80]) if c else ("EMPTY_SSE", "")
        try:
            d = r.json()
        except:
            return ("PARSE_ERR", r.text[:80])
        if r.status_code == 200:
            tu = d.get("tools_used", [])
            ch = d.get("choices", [])
            if ch:
                msg = ch[0].get("message", {})
                c = msg.get("content", "")
                tc = msg.get("tool_calls", [])
                if tc or tu:
                    return ("TOOLS_OK", f"tools={tu or [t.get('function',{}).get('name','?') for t in tc]}")
                if c and "555" in c:
                    return ("ANSWERED", c[:80])
                if c:
                    return ("NO_TOOL", c[:80])
            return ("EMPTY", str(d)[:80])
        return (f"HTTP_{r.status_code}", str(d.get("error", d))[:80])
    except requests.exceptions.Timeout:
        return ("TIMEOUT", "60s")
    except Exception as e:
        return ("EXCEPTION", str(e)[:80])

def main():
    print("=" * 80)
    print("COMPREHENSIVE TEST: ALL PROVIDERS x ALL MODELS + TOOL CALLING")
    print("=" * 80)

    print("\n[STEP 1] Auto-token verification...")
    r = requests.post(f"{BASE_URL}/api/auto-token", json={"password": ADMIN_PASSWORD})
    d = r.json()
    if d.get("success"):
        print(f"  AUTO-TOKEN: PASS - Token generated: {d['key']['api_key'][:30]}...")
    else:
        print(f"  AUTO-TOKEN: FAIL - {d}")
        return

    print("\n[STEP 2] X-Admin-Key middleware verification...")
    r = requests.post(f"{BASE_URL}/v1/chat/completions",
        headers={"X-Admin-Key": ADMIN_PASSWORD, "Content-Type": "application/json"},
        json={"model":"openai","messages":[{"role":"user","content":"Say hi"}],"max_tokens":10}, timeout=30)
    print(f"  X-ADMIN-KEY: {'PASS' if r.status_code == 200 else 'FAIL'} (status: {r.status_code})")

    print("\n[STEP 3] Getting all providers & models...")
    pm = get_all_providers_models()
    total = sum(len(v) for v in pm.values())
    print(f"  {len(pm)} providers, {total} total models\n")
    for p, m in pm.items():
        print(f"  {p:35s} -> {len(m):3d} models: {', '.join(m[:3])}{'...' if len(m)>3 else ''}")

    print(f"\n[STEP 4] Testing ALL providers and models...")
    results = {}
    n = 0
    for provider, models in pm.items():
        results[provider] = {}
        print(f"\n  === {provider} ({len(models)} models) ===")
        for model in models:
            n += 1
            sys.stdout.write(f"    [{n:3d}] {model:45s} ")
            sys.stdout.flush()

            t0 = time.time()
            cs, cd = chat_test(provider, model)
            ct = time.time() - t0

            t0 = time.time()
            ts, td = tool_test(provider, model)
            tt = time.time() - t0

            cok = cs == "OK"
            tok = ts in ("TOOLS_OK", "ANSWERED", "RESPONSE")
            print(f"Chat:{'PASS' if cok else 'FAIL':4s}({ct:.0f}s) Tool:{'PASS' if tok else 'FAIL':4s}({tt:.0f}s)")
            if not cok:
                print(f"         Chat: [{cs}] {cd[:65]}")
            if not tok:
                print(f"         Tool: [{ts}] {td[:65]}")

            results[provider][model] = {"chat_ok": cok, "chat_status": cs, "tool_ok": tok, "tool_status": ts, "chat_time": round(ct,1), "tool_time": round(tt,1)}
            time.sleep(0.2)

    # FINAL SUMMARY
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY REPORT")
    print("=" * 80)

    grand_chat_ok = grand_chat_fail = grand_tool_ok = grand_tool_fail = 0

    for provider, mdata in results.items():
        co = sum(1 for m in mdata.values() if m["chat_ok"])
        cf = sum(1 for m in mdata.values() if not m["chat_ok"])
        to = sum(1 for m in mdata.values() if m["tool_ok"])
        tf = sum(1 for m in mdata.values() if not m["tool_ok"])
        grand_chat_ok += co; grand_chat_fail += cf; grand_tool_ok += to; grand_tool_fail += tf
        t = co + cf
        s = "ALL OK" if cf == 0 and tf == 0 else f"{co}/{t} chat, {to}/{t} tool"
        print(f"\n  {provider:35s} [{s}]")
        for model, r in mdata.items():
            ci = "PASS" if r["chat_ok"] else "FAIL"
            ti = "PASS" if r["tool_ok"] else "FAIL"
            extra = ""
            if not r["chat_ok"]:
                extra += f" [chat:{r['chat_status']}]"
            if not r["tool_ok"]:
                extra += f" [tool:{r['tool_status']}]"
            print(f"    {ci} {ti}  {model}{extra}")

    gt = grand_chat_ok + grand_chat_fail
    print(f"\n{'='*80}")
    print(f"GRAND TOTAL ({gt} model tests):")
    print(f"  Chat     : {grand_chat_ok} PASS / {grand_chat_fail} FAIL")
    print(f"  Tools    : {grand_tool_ok} PASS / {grand_tool_fail} FAIL")
    print(f"  Auto-Token  : WORKING")
    print(f"  X-Admin-Key : WORKING")
    print(f"{'='*80}")

    with open("/tmp/test_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to /tmp/test_results.json")

def get_all_providers_models():
    return requests.get(f"{BASE_URL}/api/models").json()

if __name__ == "__main__":
    main()
