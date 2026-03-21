import json
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ...services.session_rag_service import SessionRetrievalService
from ..repository.session_memory_manager import SessionMemoryManager
from ..repository.workspace_registry import get_workspace_registry


class AnalyzeSessionDocumentInput(BaseModel):
    query: str = Field(..., description="用户要分析的当前会话文档问题")


def create_analyze_workspace_documents_tool(
    retrieval_service: SessionRetrievalService,
    session_manager: SessionMemoryManager,
) -> Any:
    @tool(
        "analyze_workspace_documents",
        args_schema=AnalyzeSessionDocumentInput,
        description="当需要查询、总结或对比用户在当前工作空间（会话）中上传的本地文件、财报的具体细节时，优先使用此工具。",
    )
    async def analyze_workspace_documents(query: str, config: RunnableConfig) -> str:
        configurable = (config or {}).get("configurable", {})
        workspace_id = configurable.get("workspace_id")
        if not workspace_id:
            return "System Error: Workspace ID not found in context."

        # 更新活跃时间
        get_workspace_registry().touch_workspace(workspace_id)

        collection = session_manager.get_collection_if_exists(workspace_id)
        if collection is None or collection.count() == 0:
            return json.dumps(
                [
                    {
                        "content": "当前会话尚未上传任何文档，无法查询。",
                        "document_name": "System",
                        "provider": "session_rag",
                    }
                ],
                ensure_ascii=False,
            )

        retrieved = await retrieval_service.retrieve(workspace_id=workspace_id, query=query)
        if not retrieved:
            return json.dumps(
                [
                    {
                        "content": "当前会话文档中未检索到相关信息。",
                        "document_name": "System",
                        "provider": "session_rag",
                    }
                ],
                ensure_ascii=False,
            )

        results: List[Dict[str, Any]] = []
        for item in retrieved:
            results.append(
                {
                    "content": item.chunk.content,
                    "document_name": item.chunk.document_name or "Session Document",
                    "score": item.rerank_score if item.rerank_score is not None else item.search_score,
                    "provider": "session_rag",
                }
            )
        return json.dumps(results, ensure_ascii=False)

    return analyze_workspace_documents
