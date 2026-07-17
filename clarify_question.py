"""
智能问题澄清系统（纯AI驱动）
特点：
  • 完全基于Qwen模型进行自然语言理解与生成
  • 无硬编码选项，模型自主判断缺失信息并追问
  • 支持数字/口语/简写等自由表达
  • 模型最终输出完整、流畅的精准问题
"""
import os
import sys
from typing import List, Dict, Tuple
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch


class AIQuestionClarifier:
    def __init__(self, model_name: str = "models/Qwen3-1.7B"):
        print("🔍 加载Qwen模型中（用于问题澄清）...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        self.device = next(self.model.parameters()).device
        print(f"✓ 澄清模型加载完成 (device: {self.device})")
        
        # 系统角色定义（关键：约束模型行为）
        self.system_prompt = (
            "你是一名交通监管领域专家助手，负责将用户模糊问题澄清为精准查询语句。\n"
            "工作流程：\n"
            "1. 分析用户问题缺失的关键信息（业务领域/具体场景/约束条件/信息类型）\n"
            "2. 每轮只追问1个最缺失的维度，用自然语言提问（禁止使用①②③选项）\n"
            "3. 理解用户自由回答（数字/口语/简写均可），提取关键信息\n"
            "4. 当问题足够清晰时，直接输出完整精准问题（以'【精准问题】'开头）\n"
            "5. 对话轮数≤10，避免过度追问\n"
            "业务领域知识：\n"
            "- 危化品运输：槽罐车、易燃液体、剧毒化学品、押运员\n"
            "- 道路客运：大巴、班车、旅游包车、乘客安全\n"
            "- 道路货运：货车、物流、冷链、载重限制\n"
            "- 事故案例：碰撞、追责、伤亡、调查报告\n"
            "输出规则：\n"
            "- 追问时：直接输出自然语言问题（如'您指的是危化品运输还是普通货运？'）\n"
            "- 确认完成时：以'【精准问题】'开头输出完整句子（如'【精准问题】2023年危化品槽罐车跨省运输超载的处罚标准'）\n"
            "- 禁止输出选项列表、markdown格式、思考过程"
        )
    
    def _build_chat_history(self, initial_question: str, history: List[Tuple[str, str]]) -> List[Dict]:
        """构建符合Qwen格式的对话历史"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": f"用户初始问题：{initial_question}"})
        messages.append({"role": "assistant", "content": "我将帮您澄清问题细节，请回答我的追问。"})
        
        for user_msg, ai_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})
        
        return messages
    
    def _generate_response(self, messages: List[Dict], max_new_tokens: int = 150) -> str:
        """调用Qwen生成回复"""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,  # 降低随机性，提高稳定性
                top_p=0.85,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        output = self.tokenizer.decode(
            generated_ids[0][len(model_inputs.input_ids[0]):], 
            skip_special_tokens=True
        ).strip()
        
        # 清理可能的重复前缀
        if output.startswith("assistant:"):
            output = output[len("assistant:"):].strip()
        if output.startswith("AI:"):
            output = output[len("AI:"):].strip()
        
        return output
    
    def clarify(self, initial_question: str, max_turns: int = 10) -> Tuple[str, List[Tuple[str, str]]]:
        """
        多轮AI澄清主流程
        
        返回:
            refined_question: 模型生成的精准问题（完整自然语言句子）
            dialogue_history: 对话历史 [(用户, AI), ...]
        """
        print("\n" + "="*70)
        print("💬 AI问题澄清助手（纯自然语言对话）")
        print("="*70)
        print(f"\n👤 您的原始问题：{initial_question}")
        print(f"\n💡 我将通过≤{max_turns}轮对话帮您明确问题细节")
        print("   随时输入 'quit' 退出，'done' 跳过剩余澄清")
        print("="*70 + "\n")
        
        history = []  # [(user_msg, ai_response), ...]
        refined_question = None
        
        for turn in range(1, max_turns + 1):
            # 构建对话历史
            messages = self._build_chat_history(initial_question, history)
            
            # AI生成追问或确认
            ai_response = self._generate_response(messages, max_new_tokens=120)
            
            # 检查是否已生成精准问题
            if "【精准问题】" in ai_response:
                refined_question = ai_response.split("【精准问题】", 1)[1].strip()
                print(f"✅ AI判断问题已清晰（第{turn-1}轮终止）")
                break
            
            # 输出AI追问
            print(f"\n🤖 [第{turn}/{max_turns}轮] {ai_response}")
            
            # 获取用户输入
            try:
                user_response = input("\n👤 您的回答：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 对话中断")
                sys.exit(0)
            
            # 处理控制命令
            if user_response.lower() in ['quit', 'exit', 'q']:
                print("\n👋 退出澄清流程")
                return initial_question, history
            if user_response.lower() == 'done':
                print("\n⏭️  用户主动结束澄清")
                # 让模型基于当前信息生成最终问题
                messages.append({"role": "user", "content": "请基于已有信息生成精准问题"})
                final_resp = self._generate_response(messages, max_new_tokens=100)
                if "【精准问题】" in final_resp:
                    refined_question = final_resp.split("【精准问题】", 1)[1].strip()
                else:
                    refined_question = initial_question + "（用户提前终止澄清）"
                break
            
            # 记录本轮对话
            history.append((user_response, ai_response))
        
        # 如果未提前终止，最后让模型生成精准问题
        if refined_question is None:
            print(f"\n⚠️  达到最大轮数({max_turns})，生成最终问题...")
            messages = self._build_chat_history(initial_question, history)
            messages.append({"role": "user", "content": "请基于以上对话生成精准查询问题"})
            final_resp = self._generate_response(messages, max_new_tokens=100)
            
            if "【精准问题】" in final_resp:
                refined_question = final_resp.split("【精准问题】", 1)[1].strip()
            else:
                # 降级方案：拼接原始问题+关键信息
                refined_question = initial_question + "（经多轮澄清）"
        
        # 输出最终结果
        print("\n" + "="*70)
        print("🎯 澄清完成！AI生成的精准查询问题：")
        print("="*70)
        print(f"\n🔍 {refined_question}\n")
        
        # 显示对话摘要
        print("📊 对话摘要：")
        for i, (user_msg, ai_msg) in enumerate(history, 1):
            print(f"  [Q{i}] AI: {ai_msg[:60]}...")
            print(f"  [A{i}] 您: {user_msg}")
        print(f"\n✅ 共{len(history)}轮对话 → 精准问题：{refined_question[:80]}...")
        print("="*70)
        
        return refined_question, history


# ==================== 独立运行入口 ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚌 交通监管知识库 - AI问题澄清助手（纯自然语言版）")
    print("="*70)
    print("\n✨ 特点：")
    print("   • 完全基于Qwen模型理解与生成")
    print("   • 支持自由表达（数字/口语/简写均可）")
    print("   • 模型自主追问缺失信息，非固定选项")
    print("   • 最终输出完整、流畅的精准问题")
    print("\n⚠️  注意：首次运行需加载模型（约10-20秒）")
    print("="*70)
    
    # 初始化AI澄清器
    try:
        clarifier = AIQuestionClarifier("models/Qwen3-1.7B")
    except Exception as e:
        print(f"\n❌ 模型加载失败: {str(e)}")
        print("   请确认 models/Qwen3-1.7B 目录存在且包含完整模型文件")
        sys.exit(1)
    
    # 主循环：支持连续澄清多个问题
    while True:
        try:
            print("\n" + "-"*70)
            raw_question = input("\n❓ 请输入您的问题（输入 'exit' 退出程序）：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break
        
        if raw_question.lower() in ['exit', 'quit', 'q']:
            print("\n👋 再见！")
            break
        
        if not raw_question:
            print("⚠️  问题不能为空，请重新输入")
            continue
        
        # 执行AI澄清对话
        refined_question, history = clarifier.clarify(raw_question, max_turns=8)
        
        # 询问是否继续
        print("\n" + "-"*70)
        cont = input("\n🔄 是否澄清新问题？(y/n)：").strip().lower()
        if cont not in ['y', 'yes', '是', '']:
            print("\n👋 感谢使用，再见！")
            break
        