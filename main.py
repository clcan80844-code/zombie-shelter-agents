import json
import os
import time
from memory_system import MemoryManager
from openai import OpenAI
from schema import AgentDecision, AgentReflection, DMEvaluation

# API 配置
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
memory_mgr = MemoryManager()


# ---------------- 1. API 结构化调用封装 (Pydantic 强校验) ----------------
def call_llm_structured(system_prompt: str, user_content: str, response_model):
    """强制要求 LLM 输出符合 Pydantic Model 的结构化 JSON"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY。请在环境变量中配置密钥后再运行命令行模拟器。"
        )

    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=60.0)
    prompt = (
        f"{system_prompt}\n\n"
        f"【输出格式要求】: 请必须输出严格符合以下 JSON Schema 的合法 JSON 对象，不要添加 markdown 标记之外的废话：\n"
        f"{json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)}"
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            raw_json = response.choices[0].message.content.strip()
            return response_model.model_validate_json(raw_json)
        except Exception as e:
            print(
                f"⚠️ [结构化解析失败，重试 {attempt+1}/3]: {e}"
            )
            time.sleep(1)

    raise ValueError("LLM 输出结构多次校验失败！")


# ---------------- 2. Agent 决策与发言 ----------------
def get_agent_turn(
    agent_name: str, persona_path: str, env_data: dict, chat_history: list
) -> AgentDecision:
    with open(persona_path, "r", encoding="utf-8") as f:
        persona = f.read().strip()

    # 注入信息不对称：公共环境 + Agent 的私密记忆
    private_memory_text = memory_mgr.get_agent_memory_text(agent_name)

    env_text = (
        f"【第 {env_data['day']} 天公共环境】\n"
        f"天气: {env_data['weather']} | 资源: 罐头 {env_data['resources']['food_cans']}个, 水 {env_data['resources']['water_bottles']}瓶, 子弹 {env_data['resources']['bullets']}发\n"
        f"威胁: {env_data['external_threat']} | 近期事件: {env_data['recent_event']}"
    )

    system_prompt = (
        f"你是末日避难所中的【{agent_name}】。\n"
        f"人设背景：\n{persona}\n\n"
        f"【你的私密内心世界（其他人完全不可见）】:\n{private_memory_text}\n\n"
        f"【物理规则提醒】：每人每天实打实消耗 1 罐头和 1 瓶水！资源归零所有人都会死！\n"
        f"请结合你的内心动机、对其他人的信任度及私密计划，做出今天的表达和行动选择。"
    )

    formatted_chat = "\n".join(
        [
            f"{speaker} ({action}): {speech}"
            for speaker, action, speech in chat_history
        ]
    )
    user_content = f"{env_text}\n\n【今日之前的群聊发言】:\n{formatted_chat if formatted_chat else '（今天讨论刚开始）'}"

    return call_llm_structured(system_prompt, user_content, AgentDecision)


# ---------------- 3. DM 物理世界结算 ----------------
def run_dm_evaluation(
    env_data: dict, daily_actions: list
) -> DMEvaluation:
    actions_summary = "\n".join(
        [
            f"- {name}: 进行了 [{act.action_type}] 行动，目标是 {act.action_target}。内心想法: '{act.private_thought}'"
            for name, act in daily_actions
        ]
    )

    system_prompt = (
        "你是一个严谨的末日游戏物理引擎兼 DM。\n"
        "你需要根据 Agent 今天的实际行动(action_type)与目标，评估今天对物理世界产生的影响。\n"
        "判定规则：\n"
        "1. SEARCH: 若外出搜寻且成功，可获得 1-2 食物或水。若无外出，搜获为 0。\n"
        "2. STEAL_RESOURCE: 如果有人偷窃，可产生 private_discoveries (私下隐藏物资)。\n"
        "3. GUARD/REST: 无额外物理产出。\n"
        "4. 输出严格符合 DMEvaluation Schema 的 JSON。"
    )

    user_content = f"昨天环境:\n{json.dumps(env_data, ensure_ascii=False)}\n\n今天 Agent 的实际动作与内心独白:\n{actions_summary}"
    return call_llm_structured(system_prompt, user_content, DMEvaluation)


# ---------------- 4. 夜间反思与长期记忆更新 ----------------
def run_night_reflection(
    agent_name: str, env_data: dict, daily_chat: list, dm_eval: DMEvaluation
):
    system_prompt = (
        f"夜深了，你是【{agent_name}】。请对今天发生的事进行深夜反思与规划。\n"
        f"回顾今天大家的言行、信任变动，并规划你明天的暗地打算。\n"
        f"输出格式严格符合 AgentReflection Schema。"
    )

    user_content = (
        f"今日公开事件: {dm_eval.recent_event}\n"
        f"今日物理结局: 搜寻得到食物+{dm_eval.food_found}, 水+{dm_eval.water_found}\n"
        f"你的私人发现/私藏: {dm_eval.private_discoveries.get(agent_name, '无')}"
    )

    reflection: AgentReflection = call_llm_structured(
        system_prompt, user_content, AgentReflection
    )

    # 写入长期记忆库
    memory_mgr.update_reflection(
        agent_name,
        {
            "summary_of_day": f"第{env_data['day']}天：{reflection.summary_of_day}",
            "trust_updates": reflection.trust_updates,
            "tomorrow_plan": reflection.tomorrow_plan,
        },
    )


# ---------------- 5. 核心游戏循环与规则引擎 ----------------
def run_simulation():
    print("==========================================================")
    print("   🤖 Stanford Agent 级别: 具有独立记忆与私密行动的避难所模拟   ")
    print("==========================================================\n")

    agents = [
        ("供给官_艾米莉", "prompts/supply_officer.txt"),
        ("探险者_雷克斯", "prompts/explorer.txt"),
        ("休闲者_波波", "prompts/slacker.txt"),
    ]

    with open("environment.json", "r", encoding="utf-8") as f:
        env = json.load(f)

    while True:
        print(f"\n******************** 第 {env['day']} 天 ********************")
        print(
            f"【资源现状】: 罐头 {env['resources']['food_cans']} | 水 {env['resources']['water_bottles']} | 子弹 {env['resources']['bullets']}"
        )
        print(f"【外部环境】: {env['weather']} | {env['external_threat']}")
        print("------------------ 白天：讨论与决策行动 ------------------\n")

        chat_history = []
        daily_actions = []

        # 1. 轮流发言与决策
        for name, persona_path in agents:
            decision: AgentDecision = get_agent_turn(
                name, persona_path, env, chat_history
            )

            # 打印公开发言与私密行动（用于控制台调试）
            print(f"💬 【{name}】公开说: “{decision.speech}”")
            print(
                f"   └─ 🧠 [内心独白]: {decision.private_thought} | 🎯 [实体 Action]: {decision.action_type} -> {decision.action_target}\n"
            )

            chat_history.append((name, decision.action_type, decision.speech))
            daily_actions.append((name, decision))
            time.sleep(1)

        # 2. 物理层结算 (Python 硬扣减 + DM 判定)
        print("🎲 [物理引擎正在结算物理规则与隐秘动作...]")
        dm_eval: DMEvaluation = run_dm_evaluation(env, daily_actions)

        # 强制扣减每日维持生存的基础物资 (3人固定消耗)
        base_food_cost = 3
        base_water_cost = 3

        new_food = (
            env["resources"]["food_cans"] - base_food_cost + dm_eval.food_found
        )
        new_water = (
            env["resources"]["water_bottles"]
            - base_water_cost
            + dm_eval.water_found
        )
        new_bullets = env["resources"]["bullets"] - dm_eval.bullets_used

        env["resources"]["food_cans"] = max(0, new_food)
        env["resources"]["water_bottles"] = max(0, new_water)
        env["resources"]["bullets"] = max(0, new_bullets)
        env["external_threat"] = dm_eval.external_threat
        env["recent_event"] = dm_eval.recent_event

        # 3. 夜间反思与长期记忆写回
        print("\n🌙 [进入夜间：Agent 正在独立反思、更新关系矩阵与制定明日暗地计划...]")
        for name, _ in agents:
            run_night_reflection(name, env, chat_history, dm_eval)

        # 4. 判定终止条件 (Game Over Check)
        next_day = env["day"] + 1
        if new_water <= 0 or new_food <= 0:
            print("\n==================================================")
            print(
                f"💀【Game Over】第 {next_day} 天，物资断绝，避难所陷入绝境！系统停止运行。"
            )
            print("==================================================")
            break
        elif next_day > 5:
            print("\n==================================================")
            print(
                f"🎉【Victory】第 {next_day} 天，幸存者们凭智慧和策略等到了救援！"
            )
            print("==================================================")
            break

        env["day"] = next_day
        with open("environment.json", "w", encoding="utf-8") as f:
            json.dump(env, f, ensure_ascii=False, indent=2)

        time.sleep(2)


if __name__ == "__main__":
    run_simulation()
