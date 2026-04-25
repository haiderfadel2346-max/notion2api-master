"""
tool_emulation.py - OpenAI Function Calling Emulation

Notion AI doesn't support native tool/function calling.
This module emulates it by:
1. Converting tool definitions into system prompt instructions
2. Parsing <tool_call> blocks from model text output
3. Formatting parsed calls into OpenAI tool_calls response format
"""

import json
import re
from typing import Any, Optional

from app.logger import logger

# Regex to extract <tool_call> JSON blocks from model output
_RE_TOOL_CALL = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

# Fallback: also match ```tool_call blocks
_RE_TOOL_CALL_FENCE = re.compile(
    r"```tool_call\s*\n(\{.*?\})\s*\n```",
    re.DOTALL,
)


def build_tools_system_prompt(tools: list[dict[str, Any]]) -> str:
    """
    Convert OpenAI tools array into a text prompt the model can understand.

    Args:
        tools: OpenAI-format tools list
            [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]

    Returns:
        System prompt text instructing the model how to call tools.
    """
    if not tools:
        return ""

    tool_descriptions = []
    for i, tool in enumerate(tools, 1):
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "No description")
        params = func.get("parameters", {})

        params_text = ""
        properties = params.get("properties", {})
        required = params.get("required", [])

        if properties:
            param_lines = []
            for pname, pinfo in properties.items():
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                req_mark = " (required)" if pname in required else " (optional)"
                param_lines.append(f"    - {pname}: {ptype}{req_mark} — {pdesc}")
            params_text = "\n".join(param_lines)

        tool_descriptions.append(
            f"{i}. **{name}**: {desc}"
            + (f"\n   Parameters:\n{params_text}" if params_text else "")
        )

    tools_list = "\n\n".join(tool_descriptions)

    return f"""You have access to the following tools/functions. When you need to call a tool, output a <tool_call> block with valid JSON:

<tool_call>
{{"name": "function_name", "arguments": {{"param1": "value1"}}}}
</tool_call>

Available tools:

{tools_list}

Rules:
- You can call multiple tools by outputting multiple <tool_call> blocks.
- If no tool call is needed, just respond normally without any <tool_call> blocks.
- The "arguments" field must be a valid JSON object matching the tool's parameters.
- Always include required parameters."""


def parse_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """
    Parse tool calls from model text output.

    Returns:
        Tuple of (tool_calls_list, remaining_text):
        - tool_calls_list: List of parsed tool calls in OpenAI format
        - remaining_text: Text with tool_call blocks removed
    """
    if not text:
        return [], ""

    tool_calls = []
    call_id_counter = 0

    # Try primary format: <tool_call>...</tool_call>
    for match in _RE_TOOL_CALL.finditer(text):
        raw_json = match.group(1)
        parsed = _try_parse_tool_json(raw_json)
        if parsed:
            call_id_counter += 1
            tool_calls.append(_format_tool_call(parsed, f"call_{call_id_counter}"))

    # Fallback: ```tool_call ... ```
    if not tool_calls:
        for match in _RE_TOOL_CALL_FENCE.finditer(text):
            raw_json = match.group(1)
            parsed = _try_parse_tool_json(raw_json)
            if parsed:
                call_id_counter += 1
                tool_calls.append(_format_tool_call(parsed, f"call_{call_id_counter}"))

    # Remove tool_call blocks from text
    remaining = _RE_TOOL_CALL.sub("", text)
    remaining = _RE_TOOL_CALL_FENCE.sub("", remaining)
    remaining = remaining.strip()

    if tool_calls:
        logger.info(
            "Parsed tool calls from model output",
            extra={
                "request_info": {
                    "event": "tool_calls_parsed",
                    "count": len(tool_calls),
                    "names": [tc["function"]["name"] for tc in tool_calls],
                }
            },
        )

    return tool_calls, remaining


def _try_parse_tool_json(raw: str) -> Optional[dict[str, Any]]:
    """Try to parse a tool call JSON string."""
    try:
        data = json.loads(raw.strip())
        if isinstance(data, dict) and "name" in data:
            return data
    except json.JSONDecodeError:
        pass
    return None


def _format_tool_call(parsed: dict[str, Any], call_id: str) -> dict[str, Any]:
    """Format parsed JSON into OpenAI tool_call structure."""
    arguments = parsed.get("arguments", {})
    if isinstance(arguments, dict):
        arguments_str = json.dumps(arguments, ensure_ascii=False)
    else:
        arguments_str = str(arguments)

    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": str(parsed.get("name", "")),
            "arguments": arguments_str,
        },
    }


def format_tool_messages_as_text(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert 'tool' role messages into 'user' role text messages
    that the model can understand (since Notion AI has no tool role).

    Also converts 'function' role messages (legacy format).
    """
    converted = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role in ("tool", "function"):
            # Convert tool result into user-readable text
            tool_call_id = msg.get("tool_call_id", "")
            name = msg.get("name", "")

            result_text = f"[Tool result"
            if name:
                result_text += f" for {name}"
            if tool_call_id:
                result_text += f" (id: {tool_call_id})"
            result_text += f"]: {content}"

            converted.append({
                "role": "user",
                "content": result_text,
            })
        elif role == "assistant" and msg.get("tool_calls"):
            # Assistant message with tool_calls — convert to text showing what was called
            tool_calls = msg.get("tool_calls", [])
            parts = [content] if content else []
            for tc in tool_calls:
                func = tc.get("function", {})
                tc_name = func.get("name", "unknown")
                tc_args = func.get("arguments", "{}")
                parts.append(f"<tool_call>\n{{\"name\": \"{tc_name}\", \"arguments\": {tc_args}}}\n</tool_call>")
            converted.append({
                "role": "assistant",
                "content": "\n".join(parts),
            })
        else:
            converted.append(msg)

    return converted


def has_tools(req_body: Any) -> bool:
    """Check if request has tools defined."""
    tools = getattr(req_body, "tools", None)
    return bool(tools and len(tools) > 0)
