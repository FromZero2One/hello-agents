# test_simple_agent.py
import pytest
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, ToolRegistry
from hello_agents.tools import CalculatorTool
from my_simple_agent import MySimpleAgent

# 加载环境变量
load_dotenv()

# 指定最大迭代次数
max_tool_iterations = 10


@pytest.fixture(scope="module")
def llm():
    """模块级共享LLM实例：pytest收集阶段不会触发LLM初始化，避免收集过慢"""
    return HelloAgentsLLM()


def test_basic_conversation(llm):
    """测试1：基础对话Agent（无工具）"""
    print("=== 测试1：基础对话 ===")
    basic_agent = MySimpleAgent(
        name="基础助手",
        llm=llm,
        system_prompt="你是一个友好的AI助手，请用简洁明了的方式回答问题。"
    )

    response1 = basic_agent.run("你好，请介绍一下自己", max_tool_iterations=max_tool_iterations)
    print(f"基础对话响应: {response1}\n")

    assert response1 is not None and len(response1) > 0


def test_tool_enhanced_conversation(llm):
    """测试2：工具增强对话"""
    print("=== 测试2：工具增强对话 ===")
    tool_registry = ToolRegistry()
    calculator = CalculatorTool()
    tool_registry.register_tool(calculator)

    enhanced_agent = MySimpleAgent(
        name="增强助手",
        llm=llm,
        system_prompt="你是一个智能助手，可以使用工具来帮助用户。",
        tool_registry=tool_registry,
        enable_tool_calling=True
    )

    response2 = enhanced_agent.run("请帮我计算 15 * 8 + 32", max_tool_iterations=max_tool_iterations)
    print(f"工具增强响应: {response2}\n")

    assert response2 is not None and len(response2) > 0


def test_stream_response(llm):
    """测试3：流式响应"""
    print("=== 测试3：流式响应 ===")
    basic_agent = MySimpleAgent(
        name="基础助手",
        llm=llm,
        system_prompt="你是一个友好的AI助手，请用简洁明了的方式回答问题。"
    )

    print("流式响应: ", end="")
    chunks = []
    for chunk in basic_agent.stream_run("请解释什么是人工智能", max_tool_iterations=max_tool_iterations):
        chunks.append(chunk)  # 内容已在stream_run中实时打印
    print()

    # 确认确实收到了流式输出
    assert len(chunks) > 0


def test_dynamic_tool_management(llm):
    """测试4：动态工具管理"""
    print("\n=== 测试4：动态工具管理 ===")
    basic_agent = MySimpleAgent(
        name="基础助手",
        llm=llm,
        system_prompt="你是一个友好的AI助手，请用简洁明了的方式回答问题。"
    )

    print(f"添加工具前: {basic_agent.has_tools()}")
    assert basic_agent.has_tools() is False

    calculator = CalculatorTool()
    basic_agent.add_tool(calculator)
    print(f"添加工具后: {basic_agent.has_tools()}")
    assert basic_agent.has_tools() is True

    tools = basic_agent.list_tools()
    print(f"可用工具: {tools}")
    assert calculator.name in tools

    # 查看对话历史
    print(f"\n对话历史: {len(basic_agent.get_history())} 条消息")


def test_tool_calling(llm):
    """测试5：动态添加工具后的工具调用"""
    print("=== 测试5：工具调用 ===")
    basic_agent = MySimpleAgent(
        name="基础助手",
        llm=llm,
        system_prompt="你是一个友好的AI助手，请用简洁明了的方式回答问题。"
    )
    # 动态添加工具后实际调用
    basic_agent.add_tool(CalculatorTool())

    response5 = basic_agent.run("请帮我计算 15 * 8 + 32", max_tool_iterations=max_tool_iterations)
    print(f"工具调用响应: {response5}\n")

    assert response5 is not None and len(response5) > 0


if __name__ == "__main__":
    # 直接用 python 运行时执行所有测试
    _llm = HelloAgentsLLM()
    test_basic_conversation(_llm)
    test_tool_enhanced_conversation(_llm)
    test_stream_response(_llm)
    test_dynamic_tool_management(_llm)
    test_tool_calling(_llm)
    print("\n✨ 所有测试完成！")
