from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator
import json
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import adispatch_custom_event

from ..llm.factory import get_research_llm
from .states import ResearchTask, RawSearchResult
from ..mcp_connector.tools import get_dynamic_mcp_tools, parse_tool_output, web_search_tool
from .prompt.worker_prompt import (
    WORKER_SEARCH_ROUTING_SYSTEM_PROMPT,
    WORKER_SEARCH_ROUTING_USER_PROMPT,
)
from .utils import construct_messages_with_fallback
from .session_tools import create_analyze_workspace_documents_tool
from ..repository.factory import get_session_retrieval_service, get_session_manager

# State for Search Subgraph (用于包装 ReAct Agent 的输入输出)
class SearchState(TypedDict):
    goal: str
    task: ResearchTask
    search_results: Annotated[List[RawSearchResult], operator.add]

async def search_agent_node(state: SearchState, config: RunnableConfig) -> Dict[str, Any]:
    """
    使用 create_react_agent 构建搜索逻辑。
    它会自动处理工具调用循环 (Action -> Tool -> Observation)。
    """
    task = state["task"]
    session_tool = create_analyze_workspace_documents_tool(
        retrieval_service=get_session_retrieval_service(),
        session_manager=get_session_manager(),
    )
    
    # 1. 动态获取工具
    mcp_tools = await get_dynamic_mcp_tools()
    
    if not mcp_tools:
        # 优化日志：只有当任务涉及企业知识但 MCP 不可用时才输出，或者简化输出
        tools = [session_tool, web_search_tool]
    else:
        tools = [session_tool] + mcp_tools + [web_search_tool]
        
    llm = get_research_llm()
    
    # 2. 优先从 Langfuse 获取搜索路由提示词
    context_vars = {
        "goal": state.get("goal", "未知研究目标"),
        "query": task.query,
        "intent": task.intent,
    }
    messages, _ = construct_messages_with_fallback(
        "worker/search-routing",
        context_vars,
    )
    
    # 3. 创建并调用 ReAct Agent
    # 在较新版本的 langgraph 中，参数名为 state_modifier
    # 如果环境版本较老，可以不使用 modifier，而是将 SystemMessage 放到初始对话中
    agent = create_react_agent(llm, tools=tools)
    
    # 初始化对话，让模型开始搜索，同时传递系统提示词
    if messages is None:
        system_msg = WORKER_SEARCH_ROUTING_SYSTEM_PROMPT
        user_msg = WORKER_SEARCH_ROUTING_USER_PROMPT.format(
            goal=context_vars["goal"],
            query=task.query,
            intent=task.intent,
        )
        initial_messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ]
    else:
        initial_messages = messages
    
    invoke_config = dict(config or {})
    invoke_config.setdefault("recursion_limit", 60)
    try:
        response = await agent.ainvoke({"messages": initial_messages}, config=invoke_config)
    except GraphRecursionError:
        await adispatch_custom_event(
            "worker_progress",
            {
                "task_id": task.id,
                "title": task.title,
                "status": "researching",
                "message": "搜索子图达到递归上限，已提前结束本轮搜索并返回当前可用结果。",
            },
            config=config,
        )
        return {
            "search_results": [
                {
                    "content": f"Search recursion limit reached for query: {task.query}",
                    "document_name": "System",
                    "url": None,
                    "score": None,
                    "provider": "system",
                }
            ]
        }
    messages = response.get("messages", [])
    
    # 4. 从消息历史中提取工具调用并记录日志
    new_results = []
    found_tool_msg = False
    
    # 提取所有工具调用信息用于日志
    tool_name_by_call_id: Dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                t_name = tc["name"]
                t_args = tc["args"]
                tc_id = tc.get("id")
                if tc_id:
                    tool_name_by_call_id[tc_id] = t_name
                # 映射工具友好名称
                friendly_name = "会话文档检索" if t_name == "analyze_workspace_documents" else \
                                "互联网搜索" if t_name in ["web_search", "web_search_tool"] else f"企业知识库 ({t_name})"
                
                await adispatch_custom_event(
                    "worker_progress", 
                    {
                        "task_id": task.id,
                        "title": task.title,
                        "status": "researching",
                        "message": f"正在使用 [{friendly_name}] 查找: {t_args.get('query', task.query)}"
                    },
                    config=config
                )
        
        if isinstance(msg, ToolMessage):
            # 记录工具执行结果状态
            t_name = tool_name_by_call_id.get(msg.tool_call_id, "未知工具")
            friendly_name = "会话文档检索" if t_name == "analyze_workspace_documents" else \
                            "互联网搜索" if t_name in ["web_search", "web_search_tool"] else f"企业知识库 ({t_name})"
            
            parsed = parse_tool_output(msg.content)
            # 过滤掉系统提示信息（如“未找到相关资料”、“尚未上传文档”等），只计算真实业务结果
            business_results = [r for r in parsed if r.get("document_name") not in ["System", "Workspace Memory"]]
            res_count = len(business_results)
            
            if res_count > 0:
                log_res = f"[{friendly_name}] 找到 {res_count} 条相关资料。"
            else:
                # 如果业务结果为空，说明全是系统提示或真的没找到
                log_res = f"[{friendly_name}] 未找到直接相关的参考资料。"
                
            await adispatch_custom_event(
                "worker_progress", 
                {
                    "task_id": task.id,
                    "title": task.title,
                    "status": "researching",
                    "message": log_res
                },
                config=config
            )

    # 5. 从消息历史中提取最后一轮工具调用的结果
    # 逻辑：从后往前找，提取最后一段连续的 ToolMessage
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            found_tool_msg = True
            # 将工具返回的字符串解析为标准格式
            parsed = parse_tool_output(msg.content)
            # 由于是倒序遍历，我们需要将结果插入到列表头部以保持顺序
            new_results = parsed + new_results
        elif found_tool_msg:
            # 一旦遇到非 ToolMessage 且之前已经找到了 ToolMessage，说明这一轮工具调用提取结束
            break
            
    return {"search_results": new_results}

# Graph Construction
def get_search_subgraph():
    """
    将 ReAct Agent 包装为一个普通的 Node，放入 StateGraph 中。
    这样可以保持与外部系统的 State (SearchState) 兼容。
    """
    workflow = StateGraph(SearchState)
    
    workflow.add_node("search_agent", search_agent_node)
    workflow.add_edge(START, "search_agent")
    workflow.add_edge("search_agent", END)
    
    return workflow.compile()
