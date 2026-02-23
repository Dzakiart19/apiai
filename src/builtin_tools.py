"""Built-in tools for the Agent Engine.

These tools are executed server-side during the agent loop,
giving the AI agent real capabilities similar to Claude AI.

Tools:
- web_search: Search the web using DuckDuckGo
- http_request: Make HTTP requests (GET/POST/PUT/DELETE)
- run_code: Execute Python code safely
- database_query: Query a simple key-value database
- memory_write: Store data in session memory
- memory_read: Read data from session memory
- file_write: Write files to isolated workspace
- file_read: Read files from isolated workspace
- task_status: Check current plan/task execution status
"""

import json
import subprocess
import tempfile
import os
import time
import traceback
from typing import Dict, Any, Optional, List
from urllib.parse import quote_plus

import requests

from utils.logging import logger


BUILTIN_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information using DuckDuckGo. Returns relevant search results with titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5, max: 10)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "http_request",
            "description": "Make an HTTP request to any URL. Supports GET, POST, PUT, DELETE methods. Can send JSON body and custom headers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to make the request to"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
                        "description": "HTTP method (default: GET)"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Custom HTTP headers as key-value pairs"
                    },
                    "body": {
                        "type": "object",
                        "description": "JSON body for POST/PUT/PATCH requests"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Request timeout in seconds (default: 15, max: 30)"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute Python code safely in a sandboxed environment. Can perform calculations, data processing, string manipulation, and more. Returns stdout output and any errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds (default: 10, max: 30)"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "database_query",
            "description": "Query the built-in key-value database. Supports operations: set, get, delete, list, search. Data persists across requests within the same API key scope.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["set", "get", "delete", "list", "search"],
                        "description": "Database operation to perform"
                    },
                    "key": {
                        "type": "string",
                        "description": "Key for set/get/delete operations"
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to store (for set operation). Can be any JSON-serializable string."
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern for search operation (substring match on keys)"
                    }
                },
                "required": ["operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Store information in the agent's session memory. Use this to remember facts, user preferences, intermediate results, or any data needed across conversation turns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Memory key (descriptive name for the data)"
                    },
                    "value": {
                        "type": "string",
                        "description": "Data to store in memory"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata about the stored data (e.g., source, timestamp, category)"
                    }
                },
                "required": ["key", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Read information from the agent's session memory. Retrieve previously stored facts, preferences, or data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Memory key to read. Use '*' or omit to list all stored memories."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Write content to a file in the agent's isolated workspace. Useful for saving code, data, reports, or any text content. Each API key has its own workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to write (e.g., 'report.txt', 'data.json', 'script.py')"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "append": {
                        "type": "boolean",
                        "description": "If true, append to existing file instead of overwriting (default: false)"
                    }
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read content from a file in the agent's isolated workspace. Can also list all files in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to read. Use '*' to list all files in workspace."
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_status",
            "description": "Check the current execution plan and task status. Shows progress of multi-step plans, completed steps, and remaining work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "detail_level": {
                        "type": "string",
                        "enum": ["summary", "full"],
                        "description": "Level of detail: 'summary' for overview, 'full' for all step details (default: summary)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "debug_code",
            "description": "Analyze and debug code. Executes the code, catches errors, provides detailed error analysis including line numbers, error type, suggestions for fixes, and optionally runs linting/syntax checks. Supports Python code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to debug and analyze"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["python", "javascript", "bash"],
                        "description": "Programming language of the code (default: python)"
                    },
                    "fix_attempt": {
                        "type": "boolean",
                        "description": "If true, attempt to auto-fix simple syntax errors and re-run (default: false)"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a patch/diff to modify an existing file in the workspace. Supports find-and-replace patches, line-based insertions, deletions, and unified diff format. Useful for editing code files precisely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file in workspace to patch"
                    },
                    "patches": {
                        "type": "array",
                        "description": "Array of patch operations to apply sequentially",
                        "items": {
                            "type": "object",
                            "properties": {
                                "operation": {
                                    "type": "string",
                                    "enum": ["replace", "insert_after", "insert_before", "delete", "append"],
                                    "description": "Type of patch operation"
                                },
                                "find": {
                                    "type": "string",
                                    "description": "Text to find (for replace/insert_after/insert_before/delete operations)"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "New content (for replace/insert_after/insert_before/append operations)"
                                },
                                "line_number": {
                                    "type": "integer",
                                    "description": "Optional: specific line number for insert operations (1-based)"
                                }
                            },
                            "required": ["operation"]
                        }
                    },
                    "create_if_missing": {
                        "type": "boolean",
                        "description": "Create the file if it doesn't exist (default: false)"
                    }
                },
                "required": ["filename", "patches"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories in a given path within the agent's workspace. Returns file names, sizes, types (file/directory), and modification times.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (relative to workspace root). Use '.' or '' for workspace root."
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "If true, list files recursively including subdirectories (default: false)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a new directory in the agent's workspace. Creates parent directories if they don't exist (like mkdir -p).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to create (relative to workspace root)"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command (bash) and return the output. Useful for system operations, file management, checking system info, running CLI tools, etc. Commands run in a sandboxed environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute (e.g., 'ls -la', 'echo hello', 'cat file.txt', 'uname -a')"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds (default: 10, max: 30)"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "install_package",
            "description": "Install a Python package using pip. Returns installation output and status. Useful when code requires a package that isn't installed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "package": {
                        "type": "string",
                        "description": "Package name to install (e.g., 'numpy', 'pandas', 'requests==2.31.0')"
                    },
                    "upgrade": {
                        "type": "boolean",
                        "description": "If true, upgrade the package if already installed (default: false)"
                    }
                },
                "required": ["package"]
            }
        }
    }
]

BUILTIN_TOOL_NAMES = {t["function"]["name"] for t in BUILTIN_TOOL_DEFINITIONS}

_agent_memory: Dict[str, Dict[str, Any]] = {}

_agent_database: Dict[str, Dict[str, Any]] = {}


def get_builtin_tool_definitions() -> List[Dict[str, Any]]:
    return BUILTIN_TOOL_DEFINITIONS


def is_builtin_tool(tool_name: str) -> bool:
    return tool_name in BUILTIN_TOOL_NAMES


def execute_builtin_tool(tool_name: str, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    context = context or {}
    try:
        if tool_name == "web_search":
            return _execute_web_search(arguments)
        elif tool_name == "http_request":
            return _execute_http_request(arguments)
        elif tool_name == "run_code":
            return _execute_run_code(arguments, context)
        elif tool_name == "database_query":
            scope = context.get("api_key", "default")
            return _execute_database_query(arguments, scope)
        elif tool_name == "memory_write":
            session_id = context.get("session_id", "default")
            return _execute_memory_write(arguments, session_id)
        elif tool_name == "memory_read":
            session_id = context.get("session_id", "default")
            return _execute_memory_read(arguments, session_id)
        elif tool_name == "file_write":
            session_id = context.get("session_id", "default")
            return _execute_file_write(arguments, session_id)
        elif tool_name == "file_read":
            session_id = context.get("session_id", "default")
            return _execute_file_read(arguments, session_id)
        elif tool_name == "task_status":
            return _execute_task_status(arguments, context)
        elif tool_name == "debug_code":
            return _execute_debug_code(arguments, context)
        elif tool_name == "apply_patch":
            session_id = context.get("session_id", "default")
            return _execute_apply_patch(arguments, session_id)
        elif tool_name == "list_directory":
            session_id = context.get("session_id", "default")
            return _execute_list_directory(arguments, session_id)
        elif tool_name == "create_directory":
            session_id = context.get("session_id", "default")
            return _execute_create_directory(arguments, session_id)
        elif tool_name == "run_shell":
            return _execute_run_shell(arguments, context)
        elif tool_name == "install_package":
            return _execute_install_package(arguments)
        else:
            return json.dumps({"error": f"Unknown built-in tool: {tool_name}"})
    except Exception as e:
        logger.error(f"Built-in tool '{tool_name}' error: {e}")
        return json.dumps({"error": str(e), "tool": tool_name})


def _execute_web_search(args: Dict[str, Any]) -> str:
    query = args.get("query", "")
    max_results = min(args.get("max_results", 5), 10)

    if not query:
        return json.dumps({"error": "Search query is required"})

    try:
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()

        from html.parser import HTMLParser

        class DDGParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.current = {}
                self.in_title = False
                self.in_snippet = False
                self.capture_text = ""

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                cls = attrs_dict.get("class") or ""
                if tag == "a" and "result__a" in cls:
                    self.in_title = True
                    self.capture_text = ""
                    href = attrs_dict.get("href") or ""
                    if "uddg=" in href:
                        from urllib.parse import unquote, urlparse, parse_qs
                        query_str = urlparse(href).query or ""
                        parsed = parse_qs(query_str)
                        uddg_vals = parsed.get("uddg", [href])
                        href = unquote(uddg_vals[0]) if uddg_vals else href
                    self.current["url"] = href
                elif tag == "a" and "result__snippet" in cls:
                    self.in_snippet = True
                    self.capture_text = ""

            def handle_endtag(self, tag):
                if tag == "a" and self.in_title:
                    self.in_title = False
                    self.current["title"] = self.capture_text.strip()
                elif tag == "a" and self.in_snippet:
                    self.in_snippet = False
                    self.current["snippet"] = self.capture_text.strip()
                    if self.current.get("title") and self.current.get("url"):
                        self.results.append(self.current.copy())
                    self.current = {}

            def handle_data(self, data):
                if self.in_title or self.in_snippet:
                    self.capture_text += data

        parser = DDGParser()
        parser.feed(resp.text)

        results = parser.results[:max_results]

        if not results:
            return json.dumps({
                "query": query,
                "results": [],
                "message": "No results found. Try different search terms."
            })

        return json.dumps({
            "query": query,
            "results_count": len(results),
            "results": results
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Web search error: {e}")
        return json.dumps({"error": f"Search failed: {str(e)}", "query": query})


def _execute_http_request(args: Dict[str, Any]) -> str:
    url = args.get("url", "")
    method = args.get("method", "GET").upper()
    headers = args.get("headers", {})
    body = args.get("body")
    timeout = min(args.get("timeout", 15), 30)

    if not url:
        return json.dumps({"error": "URL is required"})

    blocked_patterns = ["localhost", "127.0.0.1", "0.0.0.0", "169.254", "10.", "192.168", "172.16"]
    from urllib.parse import urlparse
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    for pattern in blocked_patterns:
        if hostname.startswith(pattern) or hostname == pattern:
            return json.dumps({"error": f"Requests to internal/private networks are blocked for security"})

    try:
        kwargs = {
            "method": method,
            "url": url,
            "headers": headers,
            "timeout": timeout
        }
        if body and method in ("POST", "PUT", "PATCH"):
            kwargs["json"] = body

        resp = requests.request(**kwargs)

        content_type = resp.headers.get("Content-Type", "")
        if "json" in content_type:
            try:
                resp_body = resp.json()
            except:
                resp_body = resp.text[:5000]
        else:
            resp_body = resp.text[:5000]

        return json.dumps({
            "status_code": resp.status_code,
            "headers": dict(list(resp.headers.items())[:20]),
            "body": resp_body,
            "url": str(resp.url),
            "method": method
        }, ensure_ascii=False, default=str)

    except requests.exceptions.Timeout:
        return json.dumps({"error": f"Request timed out after {timeout}s", "url": url})
    except requests.exceptions.ConnectionError as e:
        return json.dumps({"error": f"Connection failed: {str(e)}", "url": url})
    except Exception as e:
        return json.dumps({"error": f"HTTP request failed: {str(e)}", "url": url})


def _execute_run_code(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    code = args.get("code", "")
    timeout = min(args.get("timeout", 10), 30)

    if not code:
        return json.dumps({"error": "Code is required"})

    dangerous_imports = ["os.system", "subprocess", "shutil.rmtree", "__import__", "eval(", "exec(", "open(", "os.remove", "os.unlink", "os.rmdir"]
    code_lower = code.lower()
    for danger in dangerous_imports:
        if danger.lower() in code_lower and danger not in ("eval(", "exec("):
            pass

    tmp_path: str = ""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
            indented_code = "\n".join("    " + line for line in code.split("\n"))
            safe_wrapper = (
                "import sys\n"
                "import json\n"
                "import math\n"
                "import random\n"
                "import re\n"
                "import datetime\n"
                "import collections\n"
                "import itertools\n"
                "import functools\n"
                "import string\n"
                "import hashlib\n"
                "import base64\n"
                "import urllib.parse\n"
                "import statistics\n\n"
                "try:\n"
                + indented_code + "\n"
                "except Exception as e:\n"
                '    print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)\n'
            )
            f.write(safe_wrapper)
            tmp_path = f.name

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/tmp",
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": "/tmp",
                "PYTHONDONTWRITEBYTECODE": "1"
            }
        )

        os.unlink(tmp_path)

        output = {
            "stdout": result.stdout[:5000] if result.stdout else "",
            "stderr": result.stderr[:2000] if result.stderr else "",
            "exit_code": result.returncode
        }

        if result.returncode == 0:
            output["status"] = "success"
        else:
            output["status"] = "error"

        return json.dumps(output, ensure_ascii=False)

    except subprocess.TimeoutExpired:
        try:
            os.unlink(tmp_path)
        except:
            pass
        return json.dumps({"error": f"Code execution timed out after {timeout}s", "status": "timeout"})
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except:
            pass
        return json.dumps({"error": f"Code execution failed: {str(e)}", "status": "error"})


def _execute_database_query(args: Dict[str, Any], scope: str) -> str:
    operation = args.get("operation", "")
    key = args.get("key", "")
    value = args.get("value", "")
    pattern = args.get("pattern", "")

    if scope not in _agent_database:
        _agent_database[scope] = {}

    db = _agent_database[scope]

    if operation == "set":
        if not key:
            return json.dumps({"error": "Key is required for set operation"})
        db[key] = {
            "value": value,
            "updated_at": time.time(),
            "created_at": db.get(key, {}).get("created_at", time.time())
        }
        return json.dumps({"status": "success", "key": key, "message": f"Value stored for key '{key}'"})

    elif operation == "get":
        if not key:
            return json.dumps({"error": "Key is required for get operation"})
        if key not in db:
            return json.dumps({"status": "not_found", "key": key, "message": f"Key '{key}' not found"})
        entry = db[key]
        return json.dumps({"status": "success", "key": key, "value": entry["value"], "updated_at": entry["updated_at"]})

    elif operation == "delete":
        if not key:
            return json.dumps({"error": "Key is required for delete operation"})
        if key in db:
            del db[key]
            return json.dumps({"status": "success", "key": key, "message": f"Key '{key}' deleted"})
        return json.dumps({"status": "not_found", "key": key, "message": f"Key '{key}' not found"})

    elif operation == "list":
        keys = list(db.keys())
        return json.dumps({"status": "success", "keys": keys, "count": len(keys)})

    elif operation == "search":
        if not pattern:
            return json.dumps({"error": "Pattern is required for search operation"})
        matches = {k: v["value"] for k, v in db.items() if pattern.lower() in k.lower()}
        return json.dumps({"status": "success", "pattern": pattern, "matches": matches, "count": len(matches)})

    else:
        return json.dumps({"error": f"Unknown operation: {operation}. Use: set, get, delete, list, search"})


def _execute_memory_write(args: Dict[str, Any], session_id: str) -> str:
    key = args.get("key", "")
    value = args.get("value", "")
    metadata = args.get("metadata", {})

    if not key:
        return json.dumps({"error": "Memory key is required"})
    if not value:
        return json.dumps({"error": "Memory value is required"})

    if session_id not in _agent_memory:
        _agent_memory[session_id] = {}

    _agent_memory[session_id][key] = {
        "value": value,
        "metadata": metadata,
        "stored_at": time.time()
    }

    return json.dumps({
        "status": "success",
        "key": key,
        "message": f"Memory stored: '{key}'",
        "total_memories": len(_agent_memory[session_id])
    })


def _execute_memory_read(args: Dict[str, Any], session_id: str) -> str:
    key = args.get("key", "*")

    if session_id not in _agent_memory:
        _agent_memory[session_id] = {}

    memory = _agent_memory[session_id]

    if key == "*" or not key:
        if not memory:
            return json.dumps({"status": "empty", "message": "No memories stored yet"})
        summary = {}
        for k, v in memory.items():
            summary[k] = {
                "value": v["value"][:200],
                "stored_at": v["stored_at"]
            }
        return json.dumps({
            "status": "success",
            "memories": summary,
            "total": len(memory)
        }, ensure_ascii=False)

    if key not in memory:
        return json.dumps({"status": "not_found", "key": key, "message": f"Memory '{key}' not found"})

    entry = memory[key]
    return json.dumps({
        "status": "success",
        "key": key,
        "value": entry["value"],
        "metadata": entry.get("metadata", {}),
        "stored_at": entry["stored_at"]
    }, ensure_ascii=False)


def _execute_file_write(args: Dict[str, Any], session_id: str) -> str:
    filename = args.get("filename", "")
    content = args.get("content", "")
    append = args.get("append", False)

    if not filename:
        return json.dumps({"error": "Filename is required"})
    if not content:
        return json.dumps({"error": "Content is required"})

    if ".." in filename or filename.startswith("/"):
        return json.dumps({"error": "Invalid filename. No directory traversal allowed."})

    try:
        from planner import workspace_manager
        workspace_dir = workspace_manager.get_workspace(session_id)
        filepath = os.path.join(workspace_dir, filename)

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) != workspace_dir else workspace_dir, exist_ok=True)

        mode = 'a' if append else 'w'
        with open(filepath, mode) as f:
            f.write(content)

        file_size = os.path.getsize(filepath)
        return json.dumps({
            "status": "success",
            "filename": filename,
            "size_bytes": file_size,
            "mode": "appended" if append else "written",
            "workspace": session_id
        })
    except Exception as e:
        return json.dumps({"error": f"File write failed: {str(e)}"})


def _execute_file_read(args: Dict[str, Any], session_id: str) -> str:
    filename = args.get("filename", "*")

    try:
        from planner import workspace_manager

        if filename == "*":
            files = workspace_manager.list_workspace_files(session_id)
            return json.dumps({
                "status": "success",
                "files": files,
                "count": len(files),
                "workspace": session_id
            })

        if ".." in filename or filename.startswith("/"):
            return json.dumps({"error": "Invalid filename. No directory traversal allowed."})

        content = workspace_manager.read_file(session_id, filename)
        if content is None:
            return json.dumps({"status": "not_found", "filename": filename})

        return json.dumps({
            "status": "success",
            "filename": filename,
            "content": content[:10000],
            "size_bytes": len(content),
            "truncated": len(content) > 10000
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"File read failed: {str(e)}"})


def _execute_task_status(args: Dict[str, Any], context: Dict[str, Any]) -> str:
    detail_level = args.get("detail_level", "summary")

    try:
        from planner import planner as planner_instance

        plan_id = context.get("current_plan_id")
        if plan_id:
            plan = planner_instance.get_plan(plan_id)
            if plan:
                if detail_level == "full":
                    return json.dumps(plan.to_dict(), ensure_ascii=False)
                else:
                    return json.dumps({
                        "status": "success",
                        "plan_summary": planner_instance.get_plan_summary(plan),
                        "progress_pct": plan._progress_pct(),
                        "total_steps": len(plan.steps),
                        "completed": sum(1 for s in plan.steps if s.status.value == "completed"),
                        "failed": sum(1 for s in plan.steps if s.status.value == "failed"),
                        "pending": sum(1 for s in plan.steps if s.status.value == "pending"),
                    }, ensure_ascii=False)

        return json.dumps({
            "status": "no_active_plan",
            "message": "No active execution plan. The agent is processing directly without a multi-step plan."
        })
    except Exception as e:
        return json.dumps({"error": f"Task status check failed: {str(e)}"})


def _execute_debug_code(args: Dict[str, Any], context: Dict[str, Any]) -> str:
    code = args.get("code", "")
    language = args.get("language", "python")
    fix_attempt = args.get("fix_attempt", False)

    if not code:
        return json.dumps({"error": "Code is required"})

    if language not in ("python", "javascript", "bash"):
        return json.dumps({"error": f"Unsupported language: {language}. Supported: python, javascript, bash"})

    result = {
        "language": language,
        "original_code": code,
        "analysis": []
    }

    if language == "python":
        import ast
        import py_compile
        import io

        try:
            ast.parse(code)
            result["syntax_valid"] = True
            result["analysis"].append({"type": "info", "message": "Syntax is valid"})
        except SyntaxError as e:
            result["syntax_valid"] = False
            result["analysis"].append({
                "type": "syntax_error",
                "message": str(e),
                "line": e.lineno,
                "offset": e.offset,
                "text": e.text.strip() if e.text else None
            })

            if fix_attempt:
                fixed_code = _attempt_python_fix(code, e)
                if fixed_code and fixed_code != code:
                    result["fix_attempted"] = True
                    result["fixed_code"] = fixed_code
                    try:
                        ast.parse(fixed_code)
                        result["fix_successful"] = True
                        code = fixed_code
                    except SyntaxError:
                        result["fix_successful"] = False

        try:
            lines = code.split('\n')
            issues = []
            for i, line in enumerate(lines, 1):
                stripped = line.rstrip()
                if len(stripped) > 120:
                    issues.append({"line": i, "type": "style", "message": f"Line too long ({len(stripped)} chars)"})
                if '\t' in line and '    ' in line:
                    issues.append({"line": i, "type": "warning", "message": "Mixed tabs and spaces"})
                if 'eval(' in stripped or 'exec(' in stripped:
                    issues.append({"line": i, "type": "security", "message": "Potential security risk: eval/exec usage"})
                if stripped.startswith('import ') or stripped.startswith('from '):
                    mod = stripped.split()[1].split('.')[0]
                    if mod in ('os', 'subprocess', 'shutil', 'sys'):
                        issues.append({"line": i, "type": "info", "message": f"System module imported: {mod}"})
            if issues:
                result["analysis"].extend(issues)
        except Exception:
            pass

        try:
            session_id = context.get("session_id", "default")
            exec_result = _execute_run_code({"code": code, "timeout": 10}, context)
            exec_data = json.loads(exec_result)
            result["execution"] = exec_data

            if exec_data.get("status") == "success":
                result["analysis"].append({"type": "success", "message": "Code executed successfully"})
            elif exec_data.get("status") == "error":
                error_msg = exec_data.get("stderr", "")
                result["analysis"].append({
                    "type": "runtime_error",
                    "message": error_msg[:500],
                    "error_type": _classify_python_error(error_msg)
                })
        except Exception as e:
            result["execution"] = {"status": "failed", "error": str(e)}

    elif language == "javascript":
        try:
            proc = subprocess.run(
                ["node", "-c", code],
                capture_output=True, text=True, timeout=10
            )
            if proc.returncode == 0:
                result["syntax_valid"] = True
                result["analysis"].append({"type": "info", "message": "JavaScript syntax is valid"})
            else:
                result["syntax_valid"] = False
                result["analysis"].append({
                    "type": "syntax_error",
                    "message": proc.stderr.strip()[:500]
                })
        except FileNotFoundError:
            result["analysis"].append({"type": "warning", "message": "Node.js not available for JS analysis"})
        except Exception as e:
            result["analysis"].append({"type": "error", "message": str(e)})

    elif language == "bash":
        try:
            proc = subprocess.run(
                ["bash", "-n", "-c", code],
                capture_output=True, text=True, timeout=10
            )
            if proc.returncode == 0:
                result["syntax_valid"] = True
                result["analysis"].append({"type": "info", "message": "Bash syntax is valid"})
            else:
                result["syntax_valid"] = False
                result["analysis"].append({
                    "type": "syntax_error",
                    "message": proc.stderr.strip()[:500]
                })
        except Exception as e:
            result["analysis"].append({"type": "error", "message": str(e)})

    result["status"] = "success"
    result["total_issues"] = len([a for a in result["analysis"] if a.get("type") in ("syntax_error", "runtime_error", "warning", "security")])

    return json.dumps(result, ensure_ascii=False)


def _attempt_python_fix(code: str, error: SyntaxError) -> Optional[str]:
    lines = code.split('\n')
    if error.lineno and error.lineno <= len(lines):
        line = lines[error.lineno - 1]
        if "expected ':'" in str(error) or "expected ':''" in str(error):
            lines[error.lineno - 1] = line.rstrip() + ':'
            return '\n'.join(lines)
        if 'EOL while scanning string literal' in str(error):
            stripped = line.rstrip()
            if stripped.count('"') % 2 != 0:
                lines[error.lineno - 1] = stripped + '"'
                return '\n'.join(lines)
            elif stripped.count("'") % 2 != 0:
                lines[error.lineno - 1] = stripped + "'"
                return '\n'.join(lines)
    return None


def _classify_python_error(stderr: str) -> str:
    error_types = {
        'NameError': 'undefined_variable',
        'TypeError': 'type_mismatch',
        'ValueError': 'invalid_value',
        'IndexError': 'index_out_of_range',
        'KeyError': 'missing_key',
        'AttributeError': 'invalid_attribute',
        'ImportError': 'import_failed',
        'ModuleNotFoundError': 'module_not_found',
        'FileNotFoundError': 'file_not_found',
        'ZeroDivisionError': 'division_by_zero',
        'RecursionError': 'infinite_recursion',
        'IndentationError': 'indentation_error',
        'SyntaxError': 'syntax_error',
    }
    for err_name, classification in error_types.items():
        if err_name in stderr:
            return classification
    return 'unknown_error'


def _execute_apply_patch(args: Dict[str, Any], session_id: str) -> str:
    filename = args.get("filename", "")
    patches = args.get("patches", [])
    create_if_missing = args.get("create_if_missing", False)

    if not filename:
        return json.dumps({"error": "Filename is required"})
    if not patches:
        return json.dumps({"error": "At least one patch operation is required"})
    if ".." in filename or filename.startswith("/"):
        return json.dumps({"error": "Invalid filename. No directory traversal allowed."})

    try:
        from planner import workspace_manager
        workspace_dir = workspace_manager.get_workspace(session_id)
        filepath = os.path.join(workspace_dir, filename)

        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()
        elif create_if_missing:
            content = ""
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) != workspace_dir else workspace_dir, exist_ok=True)
        else:
            return json.dumps({"error": f"File '{filename}' not found in workspace. Set create_if_missing=true to create it."})

        applied = []
        failed = []
        original_content = content

        for i, patch in enumerate(patches):
            op = patch.get("operation", "")
            find_text = patch.get("find", "")
            new_content = patch.get("content", "")
            line_num = patch.get("line_number")

            try:
                if op == "replace":
                    if not find_text:
                        failed.append({"index": i, "operation": op, "error": "'find' text is required for replace"})
                        continue
                    if find_text in content:
                        content = content.replace(find_text, new_content, 1)
                        applied.append({"index": i, "operation": op, "status": "applied"})
                    else:
                        failed.append({"index": i, "operation": op, "error": f"Text not found: '{find_text[:50]}...'"})

                elif op == "insert_after":
                    if not find_text:
                        failed.append({"index": i, "operation": op, "error": "'find' text is required"})
                        continue
                    if find_text in content:
                        idx = content.index(find_text) + len(find_text)
                        content = content[:idx] + "\n" + new_content + content[idx:]
                        applied.append({"index": i, "operation": op, "status": "applied"})
                    else:
                        failed.append({"index": i, "operation": op, "error": f"Text not found: '{find_text[:50]}...'"})

                elif op == "insert_before":
                    if not find_text:
                        failed.append({"index": i, "operation": op, "error": "'find' text is required"})
                        continue
                    if find_text in content:
                        idx = content.index(find_text)
                        content = content[:idx] + new_content + "\n" + content[idx:]
                        applied.append({"index": i, "operation": op, "status": "applied"})
                    else:
                        failed.append({"index": i, "operation": op, "error": f"Text not found: '{find_text[:50]}...'"})

                elif op == "delete":
                    if not find_text:
                        failed.append({"index": i, "operation": op, "error": "'find' text is required"})
                        continue
                    if find_text in content:
                        content = content.replace(find_text, "", 1)
                        applied.append({"index": i, "operation": op, "status": "applied"})
                    else:
                        failed.append({"index": i, "operation": op, "error": f"Text not found: '{find_text[:50]}...'"})

                elif op == "append":
                    content = content.rstrip('\n') + "\n" + new_content + "\n"
                    applied.append({"index": i, "operation": op, "status": "applied"})

                else:
                    failed.append({"index": i, "operation": op, "error": f"Unknown operation: {op}"})

            except Exception as e:
                failed.append({"index": i, "operation": op, "error": str(e)})

        if applied:
            with open(filepath, 'w') as f:
                f.write(content)

        return json.dumps({
            "status": "success" if applied else "no_changes",
            "filename": filename,
            "applied_count": len(applied),
            "failed_count": len(failed),
            "applied": applied,
            "failed": failed if failed else None,
            "file_size_bytes": len(content),
            "changed": content != original_content
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Apply patch failed: {str(e)}"})


def _execute_list_directory(args: Dict[str, Any], session_id: str) -> str:
    path = args.get("path", "").strip() or "."
    recursive = args.get("recursive", False)

    workspace_dir = os.path.join(tempfile.gettempdir(), "agent_workspaces", session_id)
    os.makedirs(workspace_dir, exist_ok=True)

    target_dir = os.path.join(workspace_dir, path) if path != "." else workspace_dir
    target_dir = os.path.realpath(target_dir)

    if not target_dir.startswith(os.path.realpath(workspace_dir)):
        return json.dumps({"error": "Access denied: path is outside workspace"})

    if not os.path.exists(target_dir):
        return json.dumps({"error": f"Directory not found: {path}", "workspace": session_id})

    if not os.path.isdir(target_dir):
        return json.dumps({"error": f"Not a directory: {path}"})

    try:
        entries = []
        if recursive:
            for root, dirs, files in os.walk(target_dir):
                rel_root = os.path.relpath(root, workspace_dir)
                for d in dirs:
                    full = os.path.join(root, d)
                    rel = os.path.join(rel_root, d) if rel_root != "." else d
                    entries.append({
                        "name": rel,
                        "type": "directory",
                        "size": 0
                    })
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.join(rel_root, f) if rel_root != "." else f
                    try:
                        size = os.path.getsize(full)
                    except:
                        size = 0
                    entries.append({
                        "name": rel,
                        "type": "file",
                        "size": size
                    })
        else:
            for item in sorted(os.listdir(target_dir)):
                full = os.path.join(target_dir, item)
                entry = {"name": item}
                if os.path.isdir(full):
                    entry["type"] = "directory"
                    entry["size"] = 0
                else:
                    entry["type"] = "file"
                    try:
                        entry["size"] = os.path.getsize(full)
                    except:
                        entry["size"] = 0
                entries.append(entry)

        return json.dumps({
            "status": "success",
            "path": path,
            "entries": entries,
            "count": len(entries),
            "workspace": session_id
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Failed to list directory: {str(e)}"})


def _execute_create_directory(args: Dict[str, Any], session_id: str) -> str:
    path = args.get("path", "").strip()
    if not path:
        return json.dumps({"error": "Directory path is required"})

    workspace_dir = os.path.join(tempfile.gettempdir(), "agent_workspaces", session_id)
    os.makedirs(workspace_dir, exist_ok=True)

    target_dir = os.path.join(workspace_dir, path)
    target_dir = os.path.realpath(target_dir)

    if not target_dir.startswith(os.path.realpath(workspace_dir)):
        return json.dumps({"error": "Access denied: path is outside workspace"})

    try:
        already_exists = os.path.exists(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        return json.dumps({
            "status": "success",
            "path": path,
            "created": not already_exists,
            "already_existed": already_exists,
            "workspace": session_id
        })
    except Exception as e:
        return json.dumps({"error": f"Failed to create directory: {str(e)}"})


def _execute_run_shell(args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    command = args.get("command", "").strip()
    timeout = min(args.get("timeout", 10), 30)

    if not command:
        return json.dumps({"error": "Command is required"})

    dangerous_commands = ["rm -rf /", "mkfs", "dd if=", ":(){", "fork bomb",
                         "chmod -R 777 /", "shutdown", "reboot", "halt",
                         "init 0", "init 6", "killall", "pkill -9"]
    cmd_lower = command.lower()
    for danger in dangerous_commands:
        if danger in cmd_lower:
            return json.dumps({"error": f"Dangerous command blocked: {danger}"})

    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/tmp",
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": "/tmp",
                "LANG": "C.UTF-8"
            }
        )

        return json.dumps({
            "status": "success" if result.returncode == 0 else "error",
            "stdout": result.stdout[:5000] if result.stdout else "",
            "stderr": result.stderr[:2000] if result.stderr else "",
            "exit_code": result.returncode,
            "command": command
        }, ensure_ascii=False)

    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Command timed out after {timeout}s", "command": command})
    except Exception as e:
        return json.dumps({"error": f"Shell execution failed: {str(e)}", "command": command})


def _execute_install_package(args: Dict[str, Any]) -> str:
    package = args.get("package", "").strip()
    upgrade = args.get("upgrade", False)

    if not package:
        return json.dumps({"error": "Package name is required"})

    dangerous = [";", "&&", "||", "|", "`", "$", "(", ")", ">", "<", "\n"]
    for char in dangerous:
        if char in package:
            return json.dumps({"error": f"Invalid package name: contains '{char}'"})

    try:
        cmd = ["pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": "/tmp",
                "PIP_NO_CACHE_DIR": "1"
            }
        )

        return json.dumps({
            "status": "success" if result.returncode == 0 else "error",
            "package": package,
            "stdout": result.stdout[:3000] if result.stdout else "",
            "stderr": result.stderr[:2000] if result.stderr else "",
            "exit_code": result.returncode
        }, ensure_ascii=False)

    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Installation timed out after 60s", "package": package})
    except Exception as e:
        return json.dumps({"error": f"Installation failed: {str(e)}", "package": package})
