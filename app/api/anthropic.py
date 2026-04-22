"""
anthropic.py - Anthropic-compatible /v1/messages endpoint

将 Anthropic Messages API 格式的请求转换为 notion2api 内部格式，
使 Claude Code 等原生使用 Anthropic SDK 的客户端可以直接接入。

Anthropic Messages API 文档：
  POST /v1/messages
  POST /v1/messages?beta=...（扩展 beta 功能，透明忽略）
"""

import json
import time
import uuid
from typing import Any, Generator, Iterable, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import is_lite_mode, is_standard_mode
from app.logger import logger
from app.model_registry import is_supported_model, list_available_models, DEFAULT_MODEL
from app.notion_client import NotionUpstreamError

router = APIRouter()


# ── Anthropic 请求 Schema ────────────────────────────────────────────────────

class AnthropicContentBlock(BaseModel):
    """Anthropic content block（文本或工具调用结果等）"""
    type: str = "text"
    text: Optional[str] = None


class AnthropicMessage(BaseModel):
    role: str  # "user" | "assistant"
    # content 可以是字符串，也可以是 content block 数组
    content: Any


class AnthropicRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    messages: List[AnthropicMessage]
    system: Optional[Any] = None          # 字符串或 content block 数组
    max_tokens: int = Field(default=8192)
    stream: bool = Field(default=False)
    temperature: Optional[float] = None
    # 其他 Anthropic 字段（thinking、tools 等）透明忽略，不报错
    model_config = {"extra": "allow"}


# ── 内容提取工具函数 ──────────────────────────────────────────────────────────

def _extract_text(content: Any) -> str:
    """将 Anthropic content 字段统一提取为纯文本字符串。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "text")
                if btype == "text":
                    parts.append(str(block.get("text", "") or ""))
                elif btype == "tool_result":
                    # 工具结果：提取 content 字段
                    inner = block.get("content", "")
                    parts.append(_extract_text(inner))
                elif btype == "tool_use":
                    # 工具调用：格式化为文本，让模型理解上下文
                    name = block.get("name", "tool")
                    inp = block.get("input", {})
                    parts.append(f"[Tool call: {name}({json.dumps(inp, ensure_ascii=False)})]")
                # image / document 等暂不支持，跳过
        return "\n".join(p for p in parts if p)
    return ""


def _extract_system(system: Any) -> str:
    """提取 system 字段为纯文本。"""
    if not system:
        return ""
    if isinstance(system, str):
        return system.strip()
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "") or "").strip())
        return "\n".join(p for p in parts if p)
    return ""


def _build_openai_style_messages(req: AnthropicRequest) -> List[dict]:
    """
    把 Anthropic 请求转换为 OpenAI 风格的 messages 列表，
    供现有 build_standard_transcript / build_lite_transcript 消费。
    """
    msgs = []

    system_text = _extract_system(req.system)
    if system_text:
        msgs.append({"role": "system", "content": system_text})

    for msg in req.messages:
        text = _extract_text(msg.content)
        # 跳过完全空白的消息，避免 Notion 端报错
        if not text.strip():
            continue
        role = msg.role if msg.role in ("user", "assistant") else "user"
        msgs.append({"role": role, "content": text})

    return msgs


# ── Anthropic SSE 格式构建 ────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _make_message_start(response_id: str, model: str) -> str:
    return _sse({
        "type": "message_start",
        "message": {
            "id": response_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })


def _make_content_block_start(index: int = 0) -> str:
    return _sse({
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "text", "text": ""},
    })


def _make_delta(text: str, index: int = 0) -> str:
    return _sse({
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    })


def _make_content_block_stop(index: int = 0) -> str:
    return _sse({"type": "content_block_stop", "index": index})


def _make_message_delta(stop_reason: str = "end_turn") -> str:
    return _sse({
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": 0},
    })


def _make_message_stop() -> str:
    return _sse({"type": "message_stop"})


# ── 流式生成器 ────────────────────────────────────────────────────────────────

def _anthropic_stream_generator(
    response_id: str,
    model: str,
    first_item: Any,
    stream_gen: Iterable[Any],
) -> Generator[str, None, None]:
    """把 Notion 内部流转换为 Anthropic SSE 格式。"""

    yield _make_message_start(response_id, model)
    yield _make_content_block_start(0)

    streamed_acc = ""
    authoritative_final = ""
    started = False

    def _iter_all():
        yield first_item
        yield from stream_gen

    try:
        for raw_item in _iter_all():
            if not isinstance(raw_item, dict):
                continue
            item_type = raw_item.get("type")

            if item_type == "final_content":
                final_text = str(raw_item.get("text", "") or "").strip()
                if final_text:
                    authoritative_final = final_text
                continue

            # thinking 和 search 在 Anthropic 格式中暂不作为独立 block 输出
            if item_type in ("thinking", "search"):
                continue

            if item_type != "content":
                continue

            chunk_text = raw_item.get("text", "")
            if not chunk_text:
                continue

            started = True
            streamed_acc += chunk_text
            yield _make_delta(chunk_text, 0)

    except Exception as exc:
        logger.error(
            "Anthropic stream generator error",
            exc_info=True,
            extra={"request_info": {"event": "anthropic_stream_error"}},
        )
        error_hint = "\n\n[上游连接中断，请稍后重试。]"
        yield _make_delta(error_hint, 0)
        streamed_acc += error_hint
    finally:
        # 如果权威最终内容与已发送内容不同，补发缺失后缀
        if authoritative_final and authoritative_final != streamed_acc:
            if not streamed_acc and authoritative_final:
                yield _make_delta(authoritative_final, 0)
            elif authoritative_final.startswith(streamed_acc):
                suffix = authoritative_final[len(streamed_acc):]
                if suffix:
                    yield _make_delta(suffix, 0)

        yield _make_content_block_stop(0)
        yield _make_message_delta("end_turn")
        yield _make_message_stop()


# ── 主路由处理器 ──────────────────────────────────────────────────────────────

@router.post("/messages", tags=["anthropic"])
async def create_message(
    request: Request,
    req_body: AnthropicRequest,
    response: Response,
):
    """
    Anthropic Messages API 兼容端点。
    接受 Claude Code、claude.ai SDK 等原生 Anthropic 格式的请求，
    内部转换为 notion2api 的 Notion 调用流程。
    """
    from app.conversation import build_standard_transcript, build_lite_transcript

    pool = request.app.state.account_pool

    # 模型验证：如果传入的 model 不被支持，降级到默认模型（Claude Code 可能传入原始 Anthropic 模型名）
    model = req_body.model
    if not is_supported_model(model):
        model = DEFAULT_MODEL
        logger.info(
            f"Unsupported model '{req_body.model}', falling back to '{model}'",
            extra={"request_info": {"event": "anthropic_model_fallback", "requested": req_body.model, "fallback": model}},
        )

    # 转换消息格式
    messages = _build_openai_style_messages(req_body)
    if not messages or not any(m["role"] == "user" for m in messages):
        raise HTTPException(status_code=400, detail="messages must contain at least one user message.")

    response_id = f"msg_{uuid.uuid4().hex}"
    max_retries = min(3, len(pool.clients))

    for attempt in range(1, max_retries + 1):
        client = None
        try:
            client = pool.get_client()

            # 构建 transcript
            if is_lite_mode():
                # Lite 模式：只取最后一条 user 消息
                last_user = next(
                    (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
                )
                transcript = build_lite_transcript(last_user, model)
            else:
                # Standard / Heavy 模式：发送完整历史
                account = {"user_id": client.user_id, "space_id": client.space_id}
                transcript = build_standard_transcript(messages, model, account)

            # 调用 Notion API
            stream_gen = client.stream_response(transcript, thread_id=None)
            first_item = next(stream_gen, None)

            if first_item is None:
                raise NotionUpstreamError("Notion upstream returned empty content.", retriable=True)

            # ── 流式响应 ──────────────────────────────────────────────────
            if req_body.stream:
                return StreamingResponse(
                    _anthropic_stream_generator(response_id, model, first_item, stream_gen),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )

            # ── 非流式响应 ────────────────────────────────────────────────
            content_parts: list[str] = []
            authoritative_final = ""

            def _iter_all():
                yield first_item
                yield from stream_gen

            for raw_item in _iter_all():
                if not isinstance(raw_item, dict):
                    continue
                itype = raw_item.get("type")
                if itype == "final_content":
                    ft = str(raw_item.get("text", "") or "").strip()
                    if ft:
                        authoritative_final = ft
                    continue
                if itype in ("thinking", "search"):
                    continue
                if itype != "content":
                    continue
                t = raw_item.get("text", "")
                if t:
                    content_parts.append(t)

            full_text = "".join(content_parts)
            # 优先使用权威最终内容
            if authoritative_final and len(authoritative_final) >= len(full_text):
                full_text = authoritative_final

            if not full_text.strip():
                raise NotionUpstreamError("Notion upstream returned empty content.", retriable=True)

            return JSONResponse(content={
                "id": response_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": full_text}],
                "model": model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": len(full_text) // 4,  # 粗略估算
                },
            })

        except NotionUpstreamError as exc:
            if client is not None and exc.retriable:
                pool.mark_failed(client)
            logger.warning(
                "Anthropic endpoint: Notion upstream failed",
                extra={
                    "request_info": {
                        "event": "anthropic_notion_upstream_failed",
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "status_code": exc.status_code,
                        "retriable": exc.retriable,
                    }
                },
            )
            if attempt == max_retries or not exc.retriable:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.error(
                "Anthropic endpoint: No available client",
                extra={"request_info": {"event": "anthropic_account_pool_unavailable", "detail": str(exc)}},
            )
            return JSONResponse(
                status_code=503,
                content={"type": "error", "error": {"type": "overloaded_error", "message": str(exc)}},
            )
        except HTTPException:
            raise
        except Exception:
            if client is not None:
                pool.mark_failed(client)
            logger.error(
                "Anthropic endpoint: Unhandled error",
                exc_info=True,
                extra={"request_info": {"event": "anthropic_unhandled_exception", "attempt": attempt}},
            )
            if attempt == max_retries:
                raise HTTPException(status_code=500, detail="Unexpected internal error.")

    raise HTTPException(status_code=503, detail="Service unavailable: all upstream retries exhausted.")
