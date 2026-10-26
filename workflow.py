from typing import Dict, Any


class WorkflowEngine:
    """轻量工作流引擎：基于状态机管理多轮业务流程"""

    def __init__(self):
        self.workflows = {
            "appointment": {
                "name": "预约到店",
                "fields": ["baby_age", "phone", "visit_time"],
                "field_prompts": {
                    "baby_age": "请问宝宝现在多大月龄啦？",
                    "phone": "麻烦留一下您的联系电话~",
                    "visit_time": "您想预约什么时间到店体验呢？"
                },
                "end_template": """
✅ 【预约确认单】
宝宝月龄：{baby_age}
联系电话：{phone}
到店时间：{visit_time}
温馨提示：请携带宝宝疫苗本，园区地址为xxx路xxx号，到店前1小时会有老师与您确认。
                """
            },
            "growth_record": {
                "name": "成长档案",
                "fields": ["baby_age", "feeding", "sleep", "performance"],
                "field_prompts": {
                    "baby_age": "请告知宝宝月龄",
                    "feeding": "请描述本周喂养情况（奶量、辅食）",
                    "sleep": "请描述本周作息睡眠情况",
                    "performance": "请描述宝宝本周行为表现与发育情况"
                },
                "end_template": """
📋 【周度成长观察记录】
宝宝月龄：{baby_age}
喂养情况：{feeding}
作息睡眠：{sleep}
行为表现：{performance}
护理建议：持续观察发育情况，按时完成月龄发育训练。
                """
            }
        }
        self.current_workflow: str = None
        self.state: Dict[str, Any] = {}
        self.current_field_idx: int = 0

    def detect_intent(self, user_input: str) -> str:
        keywords_map = {
            "appointment": ["预约", "体验", "到店", "参观", "试听"],
            "growth_record": ["档案", "成长记录", "周总结", "观察记录"]
        }
        for intent, keywords in keywords_map.items():
            if any(k in user_input for k in keywords):
                return intent
        return "chat"

    def start_workflow(self, workflow_name: str) -> str:
        self.current_workflow = workflow_name
        self.state = {}
        self.current_field_idx = 0
        first_field = self.workflows[workflow_name]["fields"][0]
        return self.workflows[workflow_name]["field_prompts"][first_field]

    def process(self, user_input: str) -> Dict[str, Any]:
        if not self.current_workflow:
            intent = self.detect_intent(user_input)
            if intent in self.workflows:
                reply = self.start_workflow(intent)
                return {"finished": False, "reply": reply, "data": None}
            return {"finished": False, "reply": None, "data": None}

        wf = self.workflows[self.current_workflow]
        current_field = wf["fields"][self.current_field_idx]
        self.state[current_field] = user_input
        self.current_field_idx += 1

        if self.current_field_idx < len(wf["fields"]):
            next_field = wf["fields"][self.current_field_idx]
            return {"finished": False, "reply": wf["field_prompts"][next_field], "data": None}

        result = wf["end_template"].format(**self.state)
        workflow_data = self.state.copy()
        self.reset()
        return {"finished": True, "reply": result, "data": workflow_data}

    def reset(self):
        self.current_workflow = None
        self.state = {}
        self.current_field_idx = 0
