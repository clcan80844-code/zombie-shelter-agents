from typing import Dict, Literal
from pydantic import BaseModel, Field


# 1. Agent 每一轮的决策输出结构
class AgentDecision(BaseModel):
    private_thought: str = Field(
        description="内心真实想法（其他人看不到，用于规划和隐秘动机）"
    )
    speech: str = Field(description="在群聊里的公开发言（80字以内）", min_length=1, max_length=240)
    action_type: Literal[
        "SEARCH",
        "JOINT_SEARCH",
        "REST",
        "GUARD",
        "STEAL_RESOURCE",
        "SHARE_SECRET",
    ] = Field(
        description="选择执行的行动：SEARCH(单独搜寻), JOINT_SEARCH(倡议/参与联合搜寻), REST(休息), GUARD(警戒), STEAL_RESOURCE(偷窃)"
    )
    action_target: str = Field(
        description="行动目标（如：超市、周围、同伴等）", min_length=1, max_length=120
    )


# 2. Agent 夜间反思与规划输出结构
class AgentReflection(BaseModel):
    summary_of_day: str = Field(description="今天发生的关键事件总结")
    trust_updates: Dict[str, int] = Field(
        description="对其他人的信任度更新，字典格式，如 {'波波': -20, '艾米莉': +10}"
    )
    tomorrow_plan: str = Field(description="明天的暗地计划")


# 3. DM 物理世界结算结构
class DMEvaluation(BaseModel):
    food_found: int = Field(default=0, ge=0, le=10, description="公开获得的食物罐头数")
    water_found: int = Field(default=0, ge=0, le=10, description="公开获得的水瓶数")
    bullets_used: int = Field(default=0, ge=0, le=6, description="消耗的子弹数")
    ammo_found: int = Field(default=0, ge=0, le=10, description="探索中找到的额外子弹数")
    private_discoveries: Dict[str, str] = Field(
        default_factory=dict,
        description="特定 Agent 的私人发现，如 {'探险者_雷克斯': '私下藏了1瓶水'}",
    )
    external_threat: str = Field(description="更新后的外部环境威胁", min_length=1, max_length=240)
    recent_event: str = Field(description="今天公开发生的主要事件", min_length=1, max_length=500)
