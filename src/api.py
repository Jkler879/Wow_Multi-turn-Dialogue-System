"""
API 服务模块 (API)

该模块基于 FastAPI 实现，作为整个系统的 HTTP 入口，对外提供 RESTful 接口。
它将用户请求转发给核心处理函数 handle_request_async，并返回最终答案。

主要功能：
    - `/chat` (POST): 接收用户输入、会话 ID 和用户 ID，返回助手回答。
    - `/health` (GET): 健康检查接口，用于服务存活探针。

依赖：
    - src.main.handle_request_async: 核心异步处理逻辑。
    - src.main.background_tasks: 全局后台任务集合，用于在应用关闭前等待任务完成。

部署：
    直接运行本文件即可启动 Uvicorn 服务器：
    python src/api.py

Author: Ke Meng
Created: 2026-03-18
Last Modified: 2026-03-18

"""

import os
import uvicorn
import asyncio
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
# 从主模块导入异步处理函数和后台任务集合
from src.main import handle_request_async, background_tasks

app = FastAPI(title="ReAct Agent API", description="多轮对话系统")


class ChatRequest(BaseModel):
    user_input: str
    session_id: str
    user_id: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    处理用户对话请求
    - **user_input**: 用户输入
    - **session_id**: 会话ID（用于区分不同对话）
    - **user_id**: 用户ID
    - **memory_short**: 是否存在短期记忆
    - **memory_long**: 是否存在长期记忆
    """
    # 直接调用异步处理函数，await 等待主流程完成（缓存、短期记忆等）
    # 长期记忆的异步任务会在后台继续执行，不会阻塞响应
    answer = await handle_request_async(request.user_input, request.session_id, user_id=request.user_id)
    return ChatResponse(answer=answer)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭前等待所有后台任务完成（最多等待30秒）"""
    if background_tasks:
        print(f"等待 {len(background_tasks)} 个后台任务完成...")
        # 等待所有后台任务，超时30秒
        done, pending = await asyncio.wait(background_tasks, timeout=30)
        if pending:
            print(f"警告：{len(pending)} 个后台任务未在30秒内完成，将被取消")
        else:
            print("所有后台任务已完成")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

