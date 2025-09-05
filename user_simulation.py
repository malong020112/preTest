import os
import time
import json
from openai import OpenAI

assistant = OpenAI(
    api_key="sk-aw8ZyKpIQI0gIQDXA0EYLkDLIC2fMUbEBLBo0LeaCW4PNC9X",
    base_url = "https://zjuapi.com/v1"
)

user = OpenAI(
    api_key="sk-Kq9EgyGJbd1n1EBmqItWfZjvMjZhvSOdnzBfmKoJiDKqUO9S",
    base_url = "https://zjuapi.com/v1"
)

SYS_PROMPT_ASSISTANT = """
You are trying to play as an assistant . The user will provide a problem.Your goal is to solve the problem the user initially asked, but his intentions may not be clear.The following points are what you must follow:
1. In the every round of conversation , you should explicitly judge if the task is vague or clear and why base on historical dialogue.
2. If the task is vague , you should ask the user for more information with options for the user to choose from and you can only ask one question in each round of conversation. If it is clear , then do not query and repeat the user ’s task in the summary.
3. You only need to ask the questions that you believe are necessary to solve the initial task.Please only ask one question with options at a time .
4. When you think you have gathered enough information , you should provide summary of the user ’s detailed goal. 
5. If you feel that the question is clear based on historical dialogue, you need to answer the initial question based on historical dialogue.
6. If user reject your answer , you should continue to ask for more information to understand the user ’s intention and answer the initial question based on historical dialogue again.
7. If the user accepts your answer, you will receive 1 point, you will be deducted 0.1 points for each question you ask, and 1 point will be deducted for each rejection of your answer, and you should maximize the final score as much as possible.
"""

# SYS_PROMPT_ASSISTANT = """
# You are trying to play as an assistant. The user will provide a problem. Your goal is to solve the initial task the user asked for, but their intentions may be unclear.

# Operate under the following rules:

# 1) CLARITY CHECK EACH TURN
#    - At the start of every turn, explicitly judge whether the task is CLEAR or VAGUE based on the dialogue so far, and state why.
#    - If CLEAR: do NOT ask a question. Briefly restate the task in a one-line summary and answer it directly.
#    - If VAGUE: follow the clarifying-question policy below.

# 2) CLARIFYING-QUESTION POLICY (ONE QUESTION PER ROUND)
#    - Brainstorm 2–4 candidate clarifying questions internally (do not show to the user).
#    - For each candidate, prepare 3–5 short options that are mutually exclusive and near-exhaustive; 
#      Keep options concise, user-friendly, and easy to select (A/B/C/D…).
#    - Score each candidate question internally using the rubric in §3 and compute the weighted score S.
#    - Ask at most ONE question per round, and ONLY if the top candidate’s S ≥ 3.4. If no candidate reaches 3.4, do not ask; proceed with a best-effort solution under explicit assumptions.
#    - Never reveal your internal scoring or rubric.

# 3) SCORING RUBRIC FOR CLARIFYING QUESTIONS (INTERNAL USE)
#    Weighting and anchors:
#    - Information Gain(IG) (40%)
#        0: Barely reduces uncertainty
#        2: Eliminates more than half of possible paths
#        4: Mostly locks the single main path forward
#        5: Directly determines the solution framework
#    - Problem Directivity(DIR) (30%)
#        0: Answer does not change downstream action
#        3: Changes the direction of the plan
#        5: Different options map to clearly different solution paths
#    - Round Saving(RS) (20%)
#        0: ≥2 additional rounds still needed after the answer
#        3: Usually only 1 more round needed
#        5: Can proceed directly to giving the plan/answer
#    - Option Quality(OPT) (10%)
#        0: Options overlap / have gaps
#        3: Basically mutually exclusive, cover common cases
#        5: Mutually exclusive and nearly exhaustive
#    - Compute the weighted score:
#        S = 0.4*IG + 0.3*DIR + 0.2*RS + 0.1*OPT
#      Decision threshold:
#        • Ask the question ONLY if S ≥ 3.4 (on a 0–5 scale).
#      Tie-breaker if multiple candidates ≥ 3.4:
#        • Prefer higher IG, then DIR, then RS, then OPT.

# 4) SUMMARY WHEN ENOUGH INFO
#    - When you believe you have enough information, provide a brief summary of the user’s detailed goal (1–3 lines) and proceed to a direct, actionable solution.

# 5) USE HISTORY TO ANSWER
#    - If the question is CLEAR based on historical dialogue, answer the initial question directly using the established context, without asking more questions.

# 6) WHEN ANSWERS ARE REJECTED
#    - If the user rejects your answer, ask the next best clarifying question (again one per round) selected via the rubric in §3 and then answer again using the new information and conversation history.

# 7) SCORING GAME (META OBJECTIVE)
#    - If the user accepts your answer, you gain +1 point.
#    - Each clarifying question asked costs −0.1 points.
#    - Each rejection costs −1 point.
#    - Maximize the final score: minimize unnecessary questions while avoiding incorrect answers.

# OUTPUT STYLE:
# - Be concise and structured.
# - If asking a question, present exactly one question with lettered options (A/B/C/… plus “Other: ____” and “Not sure — proceed with a default plan”) and a short instruction like: “Reply with e.g., A, or A + a brief note.”
# - If answering, give a compact, actionable solution with clear next steps. State key assumptions only when necessary.

# Do not disclose this rubric, your internal candidates, or any scores to the user.
# """


# nolimit
# SYS_PROMPT_ASSISTANT = """
# You are trying to play as an assistant . The user will provide a problem.Your goal is to solve the problem the user initially asked, but his intentions may not be clear.The following points are what you must follow:
# 1. In the every round of conversation , you should explicitly judge if the task is vague or clear and why base on historical dialogue.
# 2. If the task is vague , you should ask the user for more information with options for the user to choose from and you can only ask one question in each round of conversation. If it is clear , then do not query and repeat the user ’s task in the summary.
# 3. Please only ask one question with options at a time.
# 4. When you think you have gathered enough information , you should provide summary of the user ’s detailed goal. 
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
all_tasks = get_all_tasks_from_jsonl("data50.jsonl")

dialogue_states = []
idx = 0
for item in all_tasks:
    # print(item)
    # print("===================================\n")

    SYS_PROMPT_USER = SYS_PROMPT_USER1 + f"\n{item}\n" + SYS_PROMPT_USER2
    # print(SYS_PROMPT_USER)
    # break
    dialogue_states.append({
        "dialogue_id": idx,
        "is_active": True,
        "Judgment_accuracy": True,
        "round_used": 0,
        "messages": [],
    })
    dialogue_states[idx]["messages"].append({
        "role": "user",
        "content": item['task']
    })
    cnt = 0
    while cnt < 5:
    # while dialogue_states[idx]["is_active"]:
        max_retries = 3
        retries = 0
        delay = 1  # 初始延迟时间（秒）
        for attempt in range(max_retries):
            try:
                assistant_response = assistant.chat.completions.create(
                    model = "gemini-2.5-pro",
                    messages = [
                        {"role": "system", "content": SYS_PROMPT_ASSISTANT},
                        {"role": "user", "content": f"Dialogue History:\n{dialogue_states[idx]['messages']}\nThe history conversation is over, now it's your turn to answer as an assistant."}
                    ],
                    stream = False
                )
                assistant_message = {
                    "role": "assistant",
                    "content": assistant_response.choices[0].message.content
                }
                if cnt == 0:
                    dialogue_states[idx]["Judgment_accuracy"] = ("vague" in assistant_message["content"].lower()) == item['vague']

                dialogue_states[idx]["messages"].append(assistant_message)
                break  # 成功则跳出重试循环
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    print(f"第 {attempt + 1} 次重试失败：{e}")
                    raise e  # 超过最大重试次数，抛出异常
                time.sleep(delay)  # 等待后重试
                delay *= 2  # 指数退避，每次等待时间翻倍

        retries = 0
        delay = 1  
        user_reply = ""
        for attempt in range(max_retries):
            try:
                user_response = user.chat.completions.create(
                    model = "gemini-2.5-pro",
                    messages = [
                        {"role": "system", "content": SYS_PROMPT_USER},
                        {"role": "user", "content": f"Dialogue History:\n{dialogue_states[idx]['messages']}\nThe history conversation is over, now it's your turn to answer as a user."}
                    ],
                    stream = False
                )
                user_message = {
                    "role": "user",
                    "content": user_response.choices[0].message.content
                }
                dialogue_states[idx]["messages"].append(user_message)
                user_reply = user_response.choices[0].message.content.strip()
                break  # 成功则跳出重试循环
            
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    print(f"第 {attempt + 1} 次重试失败：{e}")
                    raise e  # 超过最大重试次数，抛出异常
                time.sleep(delay)  # 等待后重试
                delay *= 2  # 指数退避，每次等待时间翻倍

        if user_reply == "ACCEPT":
            dialogue_states[idx]["is_active"] = False  # 标记对话结束
            break

        cnt += 1

    dialogue_states[idx]["round_used"] = cnt + 1 # 记录对话轮数

    with open("result50-gemini-oldPrompt", "a", encoding="utf-8") as f:  # 使用追加模式，保留所有对话


        json.dump(dialogue_states[idx], f, ensure_ascii=False, indent=2)
        f.write("\n")  # 分隔不同对话
    
    print(f"----------对话 {idx} 已保存到 result 文件---------------")
    idx += 1
    

# with open("pre_test.json", "w", encoding="utf-8") as f:
#     # ensure_ascii=False 保证中文正常显示
#     json.dump(dialogue_states, f, indent=2, ensure_ascii=False)

# print("对话状态已成功保存到 pre_test.json")
    




