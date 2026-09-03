

def test_0_init():
    from openai import OpenAI
    """
    最基本的大模型调用
    """

    # 创建客户端
    _client = OpenAI(
        api_key='ollama',
        base_url='http://127.0.0.1:11434/v1',
    )
    # 调用
    response = _client.chat.completions.create(
        model="sam860/lucy:1.7b",
        messages=[{"role": "user", "content": "hello"}]
    )
    # 处理结果
    choice = response.choices[0]
    content = choice.message.content or ""
    print(f"LLM 回复：  {content}")
