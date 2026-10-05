"""
游戏内对话理解与自然语言处理系统 - 验收测试
运行方式：python -m pytest tests/test_dialogue_understanding.py -q
共 15 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.dialogue import DialogueUnderstanding


class TestDialogueUnderstanding:
    def setup_method(self):
        self.system = DialogueUnderstanding(
            {"npcs": {"blacksmith": "gruff", "innkeeper": "friendly"}}
        )

    # ---- 输入理解：分词 + 词性 ----

    def test_case_01(self):
        """分词：能把句子切分成词语"""
        tokens = self.system.tokenize("我想去铁匠铺")
        assert "我" in tokens
        assert "去" in tokens
        assert "铁匠铺" in tokens
        assert len(tokens) >= 3

    def test_case_02(self):
        """词性：能标注每个词的词性"""
        tagged = dict(self.system.pos_tag(self.system.tokenize("铁匠铺在哪里")))
        assert tagged["铁匠铺"] == "名词"
        assert tagged["哪里"] == "疑问词"

    def test_case_03(self):
        """输入理解：返回分词与词性的完整理解结果"""
        result = self.system.understand_input("我要买铁剑")
        assert result["tokens"]
        assert len(result["pos"]) == len(result["tokens"])
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result["pos"])

    # ---- 意图识别：分类 + 置信度 ----

    def test_case_04(self):
        """分类：能区分不同的意图"""
        assert self.system.intent_recognition("你好")["intent"] == "greeting"
        assert self.system.intent_recognition("铁匠铺在哪里")["intent"] == "ask_location"
        assert self.system.intent_recognition("我要买铁剑")["intent"] == "buy"
        assert self.system.intent_recognition("再见")["intent"] == "farewell"

    def test_case_05(self):
        """置信度：识别结果带 0~1 之间的置信度"""
        result = self.system.intent_recognition("我要购买药水")
        assert result["intent"] == "buy"
        assert 0.0 < result["confidence"] <= 1.0
        assert result["confidence"] >= 0.5

    def test_case_06(self):
        """分类：无法理解的输入识别为 unknown 且置信度低"""
        result = self.system.intent_recognition("zzzz qqqq")
        assert result["intent"] == "unknown"
        assert result["confidence"] == 0.0

    # ---- 实体提取：类型 + 关系 ----

    def test_case_07(self):
        """提取：能提取对话中的实体"""
        entities = self.system.entity_extraction("我想买铁剑")["entities"]
        texts = [e["text"] for e in entities]
        assert "铁剑" in texts

    def test_case_08(self):
        """类型：能区分人名、地名、物品名"""
        entities = self.system.entity_extraction("铁匠在铁匠铺卖铁剑")["entities"]
        types = {e["text"]: e["type"] for e in entities}
        assert types["铁匠"] == "person"
        assert types["铁匠铺"] == "place"
        assert types["铁剑"] == "item"

    def test_case_09(self):
        """关系：能发现实体之间的关系"""
        relations = self.system.entity_extraction("铁匠在铁匠铺卖铁剑")["relations"]
        triples = {(r["subject"], r["predicate"], r["object"]) for r in relations}
        assert ("铁匠", "位于", "铁匠铺") in triples
        assert ("铁匠", "出售", "铁剑") in triples

    # ---- 对话状态：上下文 + 槽位 ----

    def test_case_10(self):
        """状态：知道对话进行到哪里"""
        assert self.system.dialogue_state()["turn"] == 0
        self.system.understand_input("你好")
        self.system.understand_input("我要买铁剑")
        state = self.system.dialogue_state()
        assert state["turn"] == 2
        assert state["current_intent"] == "buy"

    def test_case_11(self):
        """上下文：状态记录之前聊了什么"""
        self.system.understand_input("你好")
        self.system.understand_input("我要买铁剑")
        context = self.system.dialogue_state()["context"]
        assert [turn["text"] for turn in context] == ["你好", "我要买铁剑"]
        assert context[1]["intent"] == "buy"

    def test_case_12(self):
        """槽位：跟踪需要收集的信息及其填充情况"""
        self.system.understand_input("我要买铁剑")
        state = self.system.dialogue_state()
        assert state["slots"]["item"] == "铁剑"
        assert "quantity" in state["missing_slots"]
        self.system.understand_input("来三把")
        state = self.system.dialogue_state()
        assert state["slots"]["quantity"] == "三"
        assert state["missing_slots"] == []

    # ---- 回复生成：模板 + 个性化 ----

    def test_case_13(self):
        """模板：根据对话状态用模板生成回复"""
        self.system.understand_input("铁匠铺在哪里")
        reply = self.system.response_generation()
        assert "铁匠铺" in reply

    def test_case_14(self):
        """个性化：不同NPC说话风格不一样"""
        self.system.understand_input("你好")
        assert self.system.response_generation("blacksmith") != \
               self.system.response_generation("innkeeper")

    def test_case_15(self):
        """端到端：NPC不再答非所问，回复与玩家问题相关"""
        self.system.understand_input("我想买三把铁剑")
        reply = self.system.response_generation("innkeeper")
        assert "铁剑" in reply
        assert "三" in reply


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
