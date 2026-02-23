"""Planner Module - Task Planning, Step Tracking, and Reflection Pass.

Provides Claude AI / Manus.im-like agent capabilities:
- Planner Phase: Breaks complex tasks into executable steps
- Step Tracking: Tracks status of each step (pending/running/completed/failed)
- Reflection Pass: Evaluates results before final answer
- Workspace Isolation: Per-session isolated workspace
"""

import json
import time
import uuid
import os
import shutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from utils.logging import logger


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Step:
    id: str
    description: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retries: int = 0
    max_retries: int = 2
    depends_on: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "status": self.status.value,
            "result": self.result[:500] if self.result else None,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": int((self.completed_at - self.started_at) * 1000) if self.started_at and self.completed_at else None,
            "retries": self.retries,
        }

    def mark_running(self):
        self.status = StepStatus.RUNNING
        self.started_at = time.time()

    def mark_completed(self, result: str):
        self.status = StepStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()

    def mark_failed(self, error: str):
        self.status = StepStatus.FAILED
        self.error = error
        self.completed_at = time.time()

    def can_retry(self) -> bool:
        return self.retries < self.max_retries

    def retry(self):
        self.retries += 1
        self.status = StepStatus.PENDING
        self.error = None
        self.started_at = None
        self.completed_at = None


@dataclass
class Plan:
    id: str
    goal: str
    steps: List[Step] = field(default_factory=list)
    status: str = "planning"
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    reflection: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status,
            "total_steps": len(self.steps),
            "completed_steps": sum(1 for s in self.steps if s.status == StepStatus.COMPLETED),
            "failed_steps": sum(1 for s in self.steps if s.status == StepStatus.FAILED),
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "reflection": self.reflection,
            "progress_pct": self._progress_pct(),
        }

    def _progress_pct(self) -> int:
        if not self.steps:
            return 0
        done = sum(1 for s in self.steps if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED))
        return int((done / len(self.steps)) * 100)

    def get_next_step(self) -> Optional[Step]:
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                if step.depends_on:
                    deps_met = all(
                        any(s.id == dep_id and s.status == StepStatus.COMPLETED for s in self.steps)
                        for dep_id in step.depends_on
                    )
                    if not deps_met:
                        continue
                return step
        return None

    def is_complete(self) -> bool:
        return all(s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED) for s in self.steps)

    def mark_complete(self):
        self.status = "completed"
        self.completed_at = time.time()


PLANNING_SYSTEM_PROMPT = """You are a task planner. Given a user's request, break it down into concrete executable steps.

RULES:
1. Analyze the task complexity
2. If the task is SIMPLE (can be answered directly), respond with:
{"type": "direct_answer", "reasoning": "brief explanation"}

3. If the task is COMPLEX (needs multiple tool calls or steps), respond with:
{
  "type": "plan",
  "goal": "summarized goal",
  "steps": [
    {
      "id": "step_1",
      "description": "what this step does",
      "tool_name": "tool to use (or null if reasoning step)",
      "tool_args": {"arg": "value"},
      "depends_on": []
    }
  ]
}

AVAILABLE TOOLS: {tools_list}

OUTPUT FORMAT: Valid JSON only. No markdown. No explanations outside JSON.
Keep plans concise - max 6 steps. Prefer fewer steps."""


REFLECTION_SYSTEM_PROMPT = """You are a quality evaluator. Review the task execution results and provide a brief assessment.

TASK GOAL: {goal}
EXECUTION RESULTS:
{results}

Evaluate:
1. Was the goal achieved? (yes/partially/no)
2. Are the results accurate and complete?
3. Any issues or improvements needed?

Respond with a JSON object:
{{
  "goal_achieved": "yes|partially|no",
  "confidence": 0.0-1.0,
  "summary": "brief assessment",
  "needs_retry": false,
  "retry_reason": null
}}

OUTPUT: Valid JSON only. No markdown."""


class Planner:
    def __init__(self):
        self._plans: Dict[str, Plan] = {}

    def create_plan_prompt(self, user_message: str, tools: List[Dict[str, Any]]) -> str:
        tools_list = []
        for tool in tools:
            func = tool.get("function", tool)
            name = func.get("name", "")
            desc = func.get("description", "")
            tools_list.append(f"- {name}: {desc}")

        return PLANNING_SYSTEM_PROMPT.replace("{tools_list}", "\n".join(tools_list))

    def parse_plan_response(self, response_text: str, user_message: str) -> Optional[Plan]:
        try:
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                import re
                match = re.search(r'```(?:json)?\s*([\s\S]*?)```', cleaned)
                if match:
                    cleaned = match.group(1).strip()

            data = json.loads(cleaned)

            if data.get("type") == "direct_answer":
                return None

            if data.get("type") == "plan":
                plan_id = f"plan_{uuid.uuid4().hex[:12]}"
                plan = Plan(id=plan_id, goal=data.get("goal", user_message))

                for step_data in data.get("steps", []):
                    step = Step(
                        id=step_data.get("id", f"step_{uuid.uuid4().hex[:8]}"),
                        description=step_data.get("description", ""),
                        tool_name=step_data.get("tool_name"),
                        tool_args=step_data.get("tool_args"),
                        depends_on=step_data.get("depends_on"),
                    )
                    plan.steps.append(step)

                self._plans[plan_id] = plan
                plan.status = "executing"
                return plan

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse plan response: {e}")

        return None

    def create_reflection_prompt(self, plan: Plan) -> str:
        results = []
        for step in plan.steps:
            status_icon = {"completed": "OK", "failed": "FAIL", "skipped": "SKIP"}.get(step.status.value, "?")
            result_preview = step.result[:300] if step.result else step.error or "no result"
            results.append(f"[{status_icon}] {step.description}: {result_preview}")

        results_text = "\n".join(results)
        return REFLECTION_SYSTEM_PROMPT.replace("{goal}", plan.goal).replace("{results}", results_text)

    def parse_reflection_response(self, response_text: str) -> Dict[str, Any]:
        try:
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                import re
                match = re.search(r'```(?:json)?\s*([\s\S]*?)```', cleaned)
                if match:
                    cleaned = match.group(1).strip()

            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return {
                "goal_achieved": "unknown",
                "confidence": 0.5,
                "summary": response_text[:200],
                "needs_retry": False,
            }

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        return self._plans.get(plan_id)

    def get_plan_summary(self, plan: Plan) -> str:
        lines = [f"Plan: {plan.goal}", f"Progress: {plan._progress_pct()}%"]
        for step in plan.steps:
            icon = {"pending": "[ ]", "running": "[>]", "completed": "[x]", "failed": "[!]", "skipped": "[-]"}.get(step.status.value, "[?]")
            lines.append(f"  {icon} {step.description}")
        return "\n".join(lines)


class WorkspaceManager:
    BASE_DIR = "/tmp/agent_workspaces"

    def __init__(self):
        os.makedirs(self.BASE_DIR, exist_ok=True)

    def get_workspace(self, session_id: str) -> str:
        workspace_dir = os.path.join(self.BASE_DIR, session_id)
        os.makedirs(workspace_dir, exist_ok=True)
        return workspace_dir

    def cleanup_workspace(self, session_id: str):
        workspace_dir = os.path.join(self.BASE_DIR, session_id)
        if os.path.exists(workspace_dir):
            try:
                shutil.rmtree(workspace_dir)
                logger.info(f"Cleaned up workspace: {session_id}")
            except Exception as e:
                logger.error(f"Failed to cleanup workspace {session_id}: {e}")

    def list_workspace_files(self, session_id: str) -> List[str]:
        workspace_dir = self.get_workspace(session_id)
        files = []
        for root, _, filenames in os.walk(workspace_dir):
            for f in filenames:
                rel = os.path.relpath(os.path.join(root, f), workspace_dir)
                files.append(rel)
        return files

    def write_file(self, session_id: str, filename: str, content: str) -> str:
        workspace_dir = self.get_workspace(session_id)
        filepath = os.path.join(workspace_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath

    def read_file(self, session_id: str, filename: str) -> Optional[str]:
        workspace_dir = self.get_workspace(session_id)
        filepath = os.path.join(workspace_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return f.read()
        return None


class LoopSupervisor:
    def __init__(self, max_iterations: int = 20, max_errors: int = 3, max_duration_sec: int = 180, tool_timeout_sec: int = 30):
        self.max_iterations = max_iterations
        self.max_errors = max_errors
        self.max_duration_sec = max_duration_sec
        self.tool_timeout_sec = tool_timeout_sec
        self.iteration = 0
        self.error_count = 0
        self.total_tool_calls: int = 0
        self.max_tool_calls: int = 50
        self.start_time = time.time()
        self.tool_call_log: List[Dict[str, Any]] = []
        self.repeated_tool_threshold = 3

    def can_continue(self) -> tuple:
        if self.iteration >= self.max_iterations:
            return False, "max_iterations_reached"
        if self.error_count >= self.max_errors:
            return False, "max_errors_reached"
        if self.total_tool_calls >= self.max_tool_calls:
            return False, "tool_spam_detected"
        elapsed = time.time() - self.start_time
        if elapsed > self.max_duration_sec:
            return False, "timeout"
        if self._detect_loop():
            return False, "infinite_loop_detected"
        return True, "ok"

    def record_iteration(self, tool_name: Optional[str] = None, tool_args: Optional[Dict] = None, success: bool = True, error: Optional[str] = None):
        self.iteration += 1
        if tool_name:
            self.total_tool_calls += 1
        if not success:
            self.error_count += 1
        self.tool_call_log.append({
            "iteration": self.iteration,
            "tool_name": tool_name,
            "tool_args_hash": hash(json.dumps(tool_args, sort_keys=True, default=str)) if tool_args else None,
            "success": success,
            "error": error,
            "timestamp": time.time(),
        })

    def _detect_loop(self) -> bool:
        if len(self.tool_call_log) < self.repeated_tool_threshold:
            return False
        recent = self.tool_call_log[-self.repeated_tool_threshold:]
        if len(set(e["tool_name"] for e in recent)) == 1 and len(set(e["tool_args_hash"] for e in recent)) == 1:
            return True

        if len(self.tool_call_log) >= 4:
            last4 = self.tool_call_log[-4:]
            names = [e["tool_name"] for e in last4]
            if names[0] == names[2] and names[1] == names[3] and names[0] != names[1]:
                return True

        if len(self.tool_call_log) >= self.repeated_tool_threshold:
            recent = self.tool_call_log[-self.repeated_tool_threshold:]
            if len(set(e["tool_name"] for e in recent)) == 1:
                if all(not e["success"] for e in recent):
                    return True

        return False

    def get_tool_timeout(self, tool_name: str) -> int:
        timeouts = {
            "run_code": 30,
            "run_shell": 30,
            "web_search": 15,
            "http_request": 20,
            "debug_code": 30,
            "install_package": 60,
        }
        return timeouts.get(tool_name, 15)

    def get_stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        return {
            "iterations": self.iteration,
            "max_iterations": self.max_iterations,
            "errors": self.error_count,
            "duration_sec": round(elapsed, 2),
            "tools_called": len([e for e in self.tool_call_log if e["tool_name"]]),
            "unique_tools": len(set(e["tool_name"] for e in self.tool_call_log if e["tool_name"])),
            "total_tool_calls": self.total_tool_calls,
            "max_tool_calls": self.max_tool_calls,
        }


def validate_tool_params(tool_call: Dict[str, Any], tools: List[Dict[str, Any]]) -> tuple:
    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("arguments", {})

    tool_def = None
    for t in tools:
        func = t.get("function", t)
        if func.get("name") == tool_name:
            tool_def = func
            break

    if not tool_def:
        return False, f"Tool '{tool_name}' not found"

    params = tool_def.get("parameters", {})
    required = params.get("required", [])
    properties = params.get("properties", {})

    for req_param in required:
        if req_param not in tool_args:
            return False, f"Missing required parameter '{req_param}' for tool '{tool_name}'"
        if tool_args[req_param] is None or (isinstance(tool_args[req_param], str) and not tool_args[req_param].strip()):
            return False, f"Required parameter '{req_param}' cannot be empty for tool '{tool_name}'"

    for arg_name, arg_value in tool_args.items():
        if arg_name in properties:
            prop_def = properties[arg_name]
            expected_type = prop_def.get("type", "")
            enum_values = prop_def.get("enum", [])

            if expected_type == "string" and not isinstance(arg_value, str):
                tool_args[arg_name] = str(arg_value)
            elif expected_type == "integer" and not isinstance(arg_value, int):
                try:
                    tool_args[arg_name] = int(arg_value)
                except (ValueError, TypeError):
                    return False, f"Parameter '{arg_name}' must be integer for tool '{tool_name}'"
            elif expected_type == "object" and not isinstance(arg_value, dict):
                return False, f"Parameter '{arg_name}' must be object for tool '{tool_name}'"

            if enum_values and arg_value not in enum_values:
                return False, f"Parameter '{arg_name}' must be one of {enum_values} for tool '{tool_name}'"

    return True, ""


planner = Planner()
workspace_manager = WorkspaceManager()
