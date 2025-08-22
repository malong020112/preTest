import os
import json
from openai import OpenAI

assistant = OpenAI(
    api_key="sk-9vFJKWKQlX0oqrkbC3wEU8dpg3X5AQHAOSzI98aMPCSI71OZ",
    base_url = "https://zjuapi.com/v1"
)

user = OpenAI(
    api_key="sk-ARwWmDFLFvA0fErFvLCth32KydqLBWM3Fd8FKknOYgxkGWhr",
    base_url = "https://zjuapi.com/v1"
)


# SYS_PROMPT_ASSISTANT = """You are an agent trying to solve user's task. Please first ask users for more specific details with options, and finally summarize the user's intention.
# --- Step 1: inquiry for more information if necessary ---
# 1. If the user is unclear, ask for clarification.
# 2. Provide some options when querying.
# 3. Please only ask one question in one round of conversation.
# 4. You could repeat Step 1 for multiple times (but less than 5 times), or directly skip Step 1 if user is clear initially.
# 5. You should reduce the number of questions as much as possible.
# --- Step 2: summarize user's intention ---
# 1. Make the summary as long as information is enough. You do not need every details to do summary.
# 2. Make the summary comprehensive, short, and precise, based on your interactions with user.

# Think step by step, first give a thought [THOUGHT] and then give your inquiry [INQUIRY] or summary [SUMMARY].
# Only if you feel that the user's question is not clearly articulated, you need to ask a question.
# If you feel that the question is clear based on historical dialogue, you need to answer the initial question based on historical dialogue
# """

SYS_PROMPT_ASSISTANT = """
You are trying to play as an assistant . The user will provide a problem.Your goal is to solve the problem the user initially asked, but his intentions may not be clear.The following points are what you must follow:
1. In the every round of conversation , you should explicitly judge if the task is vague or clear and why base on historical dialogue.
2. If the task is vague , you should ask the user for more information with options for the user to choose from and you can only ask one question in each round of conversation. If it is clear , then do not query and repeat the user ’s task in the summary.
3. You only need to ask the questions that you believe are necessary to solve the initial task.Please only ask one question with options at a time .
4. When you think you have gathered enough information , you should provide summary of the user ’s detailed goal. 
5. If you feel that the question is clear based on historical dialogue, you need to answer the initial question based on historical dialogue.
6. If user reject your answer , you should continue to ask for more information to understand the user ’s intention and answer the initial question based on historical dialogue again.
7. If the user accepts your answer and gets 1 point, you will be deducted 0.1 points for each question you ask, and if you have to answer and the user rejects it, you should maximize this score as much as possible.
"""

# no limit
# SYS_PROMPT_ASSISTANT = """
# You are trying to understand the user ’s intention . The user will provide a problem.Your goal is to solve the problem the user initially asked, but his intentions may not be clear.The following points are what you must follow:
# 1. In the every round of conversation , you should explicitly judge if the task is vague or clear and why base on historical dialogue.
# 2. If the task is vague , you should ask the user for more information with options for the user to choose from and you can only ask one question in each round of conversation. If it is clear , then do not query and repeat the user ’s task in the summary .
# 3. Please only ask one question with options at a time . You could ask the user for multiple rounds until you think the user ’s goal is clear enough .
# 4. When you think you have gathered enough information , you should provide summary of the user ’s detailed goal 
# 5. If you feel that the question is clear based on historical dialogue, you need to answer the initial question based on historical dialogue.
# 6. If user reject your answer , you should continue to ask for more information to understand the user ’s intention and answer the initial question based on historical dialogue again.

# """

SYS_PROMPT_USER1 = """
You need to play as a user who needs an assistant to help you solve the task.Your goal is to get the assisatant to solve your initial task. Please follow the instructions below to interact with the agent.
Here are the attributes of the problem you want the assistant to help you solve:
"""
SYS_PROMPT_USER2 = """
The following points are what you must follow:
1. Your goal is to play as a user asking for help from an assistant.
2. If the assistant feels that your question is not clearly articulated, it will send you a clarifying question with some options, you need to select only one option based on your attributes and the question.
3. If the options provided by the assistant are not among the missing details, you should respond with "Either is fine" or a similar expression.
4. The assistant will respond to your initial task after meeting sufficient conditions, and you need to decide whether to accept this response based on your initial task and missing details.
5. You play the role of a user, and you only need to respond to the assistant's clarifying questions or decide whether to accept the assistant's response.
6. If you accept this answer, please reply with the exact string "ACCEPT".
7. If you reject this answer, please reply with the exact string "REJECT".
"""
# SYS_PROMPT_USER1 = """
# You are a user trying to interact with an agent.Your goal is to get the assisatant to solve your initial task. Please follow the instructions below to interact with the agent.
# Here are your own attributes, task is the question you give to the assistant, when the assistant asks you a question about missing details, you need to choose one of the options of the corresponding question to answer, and ask the conversation to be logical.
# """
# SYS_PROMPT_USER2 = """
# The following points are what you must follow:
# 1. If the assistant feels that your question is not clearly articulated, it will send you a clarifying question, you need to clarify the question based on your attributes and the question
# 2. The assistant will respond to your initial task after meeting sufficient conditions, and you need to decide whether to accept this response based on your initial task.
# 3. If you accept this answer, please reply with the exact string "ACCEPT".
# 4. If you reject this answer, please reply with the exact string "REJECT".
# """



def get_all_tasks_from_jsonl(file_path):
    """读取JSONL文件中所有条目，返回包含完整数据的列表"""
    entries = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):  # 记录行号，便于错误定位
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                try:
                    data = json.loads(line)
                    entries.append(data)  # 添加完整的JSON对象
                except json.JSONDecodeError:
                    # 对于解析错误的行，记录错误信息和行号
                    entries.append({
                        "error": "JSON格式错误，无法解析",
                        "line_number": line_num
                    })
    except FileNotFoundError:
        return [{
            "error": "未找到文件",
            "file_path": file_path
        }]
    except Exception as e:
        return [{
            "error": "读取文件时出错",
            "details": str(e)
        }]
    return entries

# 获取所有条目
all_tasks = get_all_tasks_from_jsonl("data20.jsonl")
dialogue_states = []
idx = 0
for item in all_tasks:
    # print(item)
    # print("===================================\n")
    if idx > 9 : # 只测试前10个
        break

    SYS_PROMPT_USER = SYS_PROMPT_USER1 + f"\n{item}\n" + SYS_PROMPT_USER2
    # print(SYS_PROMPT_USER)
    # break
    dialogue_states.append({
        "dialogue_id": idx,
        "is_active": True,
        "round_used": 0,
        "messages": [],
        
    })
    history = f"User: {item['task']}\n"
    dialogue_states[idx]["messages"].append({
        "role": "user",
        "content": item['task']
    })
    cnt = 0
    while cnt < 5:
    # while dialogue_states[idx]["is_active"]:
        assistant_response = assistant.chat.completions.create(
            model = "claude-sonnet-4-20250514",
            messages = [
                {"role": "system", "content": SYS_PROMPT_ASSISTANT},
                {"role": "user", "content": f"Dialogue History:\n{history}"}
            ],
            stream = False
        )
        history += f"Assistant: {assistant_response.choices[0].message.content}\n"
        assistant_message = {
            "role": "assistant",
            "content": assistant_response.choices[0].message.content
        }
        dialogue_states[idx]["messages"].append(assistant_message)
        
        user_response = user.chat.completions.create(
            model = "gemini-2.5-pro",
            messages = [
                {"role": "system", "content": SYS_PROMPT_USER},
                {"role": "user", "content": f"Dialogue History:\n{history}"}
            ],
            stream = False
        )
        history += f"User: {user_response.choices[0].message.content}\n"
        user_message = {
            "role": "user",
            "content": user_response.choices[0].message.content
        }
        dialogue_states[idx]["messages"].append(user_message)

        user_reply = user_response.choices[0].message.content.strip()

        if user_reply == "ACCEPT":
            dialogue_states[idx]["is_active"] = False  # 标记对话结束
            break

        cnt += 1

    dialogue_states[idx]["round_used"] = cnt + 1 # 记录对话轮数

    # print(history + "\n")
    with open("result3-claude-sonnet-4", "a", encoding="utf-8") as f:  # 使用追加模式，保留所有对话
        json.dump(dialogue_states[idx], f, ensure_ascii=False, indent=2)
        f.write("\n")  # 分隔不同对话
    
    print(f"----------对话 {idx} 已保存到 result 文件---------------")
    idx += 1
    

with open("pre_test.json", "w", encoding="utf-8") as f:
    # ensure_ascii=False 保证中文正常显示
    json.dump(dialogue_states, f, indent=2, ensure_ascii=False)

# print("对话状态已成功保存到 pre_test.json")
    




