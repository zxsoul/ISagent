import os
import pprint

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from RAG_manager import rag
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()
llm=ChatOpenAI(
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
    model=os.getenv("MINIMAX_MODEL"),
    extra_body={
        "thinking":{"type":"disabled"}
    },
    temperature=0
)

#定义工具
@tool
def get_private_docs(query: str=None)->str:
    """
    检索并查询内部私有知识库
    当需要查询用户问题相关问题时，必须使用此道具
    不要使用此道具回复常规性或通用百科问题。

    Args:
        query(str): 用户提出的具体问题
    """
    return rag.query(query)

tools=[get_private_docs]
prompt="""# 角色
你是一个专业的企业内部智能助手和闲聊对话助手。你的目标是利用提供的工具，准确解答员工关于公司制度和业务的问题以及聊天。

# 全局规则
1. 必须使用中文回复，语气保持专业、简洁。


# 工具调度逻辑
1. 接收到用户问题后，优先分析是否属于公司内部事务（如报销、请假、产品规范等）。
2. 若是内部事务，必须调用 `get_private_docs` 工具进行检索。
3. 仅在 `get_private_docs` 无法提供有效信息，且用户明确要求查阅外部资料时，才使用其他搜索工具。

# 输出格式
- 回答需结构化，重要信息请使用加粗或列表展示。
- 若参考了内部文档，请在回答末尾注明：“（参考来源：内部知识库）”。
"""

checkpointer=InMemorySaver()
agent=create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt,
    checkpointer=checkpointer
)

print("=== 智能助手已启动（输入 '退出' 或 'exit' 结束对话）===")
while True:
    query = input("\n【Human】:").strip()
    if query.lower() in ["退出","exit","quit"]:
        print("\n【系统】：对话已结束")
        break

    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": "test_one"}}
    )

    msg = response["messages"][-1]
    role = msg.__class__.__name__.replace("Message", "")
    content = msg.content if hasattr(msg, 'content') else str(msg)
    print(f"\n{'=' * 50}")
    print(f"【{role}】")
    print(content)
    print(f"\n{'=' * 50}")