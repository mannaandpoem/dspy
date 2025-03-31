import json
import os

import dspy
from bbh import read_jsonl_bbh, calculate_score_bbh  # 从 bbh.py 导入相关功能
from dspy.clients.base_lm import GLOBAL_HISTORY
from dspy.evaluate import Evaluate
from dspy.teleprompt import MIPROv2

# 1. 配置语言模型
api_key = "..."
base_url = "..."

# 实例化两个不同的语言模型
prompt_lm = dspy.LM('gpt-4o-mini', api_key=api_key, base_url=base_url)  # 用于生成提示的模型
task_lm = dspy.LM('gpt-4o-mini', api_key=api_key, base_url=base_url)  # 用于执行任务的模型

# 配置默认的语言模型（这里仍使用 task_lm 作为全局默认）
dspy.configure(lm=task_lm)  # BBH 不需要检索模型，因此只配置语言模型


# 2. 定义一个简单的 BBH 问答模块
class SimpleBBHQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        prediction = self.generate_answer(question=question)
        # 确保输出格式与 BBH 评估兼容（提取纯文本答案）
        return dspy.Prediction(answer=prediction.answer)


# 3. 加载 BBH 数据
train_file_path = "bbh_train.jsonl"  # 训练集路径
val_file_path = "bbh_validation.jsonl"  # 验证集路径

# 读取数据并适配字段名（"input" -> "question", "target" -> "answer"）
train_examples = read_jsonl_bbh(train_file_path)
val_examples = read_jsonl_bbh(val_file_path)

# 将数据转换为 DSPy 格式
trainset = [dspy.Example(question=x["question"], answer=x["answer"]).with_inputs('question') for x in train_examples]
valset = [dspy.Example(question=x["question"], answer=x["answer"]).with_inputs('question') for x in val_examples]


# 4. 定义评估指标（使用 bbh.py 中的 calculate_score_bbh）
def bbh_exact_match_metric(example, pred, trace=None):
    score, _ = calculate_score_bbh(ground_truth=example.answer, prediction=pred.answer)
    return score


evaluate = Evaluate(devset=valset, metric=bbh_exact_match_metric, num_threads=4, display_progress=True)

# 5. 初始化程序
program = SimpleBBHQA()

# 6. 运行 MIPROv2 优化
teleprompter = MIPROv2(
    prompt_model=prompt_lm,  # 使用 gpt-4o 作为 prompt_model
    task_model=task_lm,  # 使用 gpt-4o-mini 作为 task_model
    metric=bbh_exact_match_metric,
    # num_candidates=1,  # 生成2个候选指令和few-shot示例
    # init_temperature=1.0,
    verbose=True
)

# teleprompter.auto = None  # auto: Optional[Literal["light", "medium", "heavy"]] = "medium"
teleprompter.auto = "light"  # auto: Optional[Literal["light", "medium", "heavy"]] = "medium"
# ==> STEP 1: BOOTSTRAP FEWSHOT EXAMPLES <==
# 在 compile 方法中，MIPROv2 首先通过 _bootstrap_fewshot_examples 方法生成少样本示例（few-shot examples），
# 这些示例将用于后续的指令生成和优化。
# ==> STEP 2: PROPOSE INSTRUCTION CANDIDATES <==
# 接着，MIPROv2 调用 _propose_instructions 方法，利用 prompt_model 生成一组指令候选（instruction candidates），
# 这些候选基于训练数据和生成的少样本示例。
# ==> STEP 3: FINDING OPTIMAL PROMPT PARAMETERS <==
# 最后，MIPROv2 通过 _optimize_prompt_parameters 方法，使用 task_model 在验证集上评估不同指令和示例组合，
# 并通过贝叶斯优化找到最优的参数组合。
optimized_program = teleprompter.compile(
    program,
    trainset=trainset,
    valset=valset,
    minibatch=True,  # 启用 minibatch，使用默认值
    requires_permission_to_run=False,  # 跳过用户确认步骤以便脚本自动化运行
    # minibatch_size=1,
    # minibatch_full_eval_steps=1,
    # num_trials=1,
)

# 7. 评估优化前后的程序
baseline_score = evaluate(program)
optimized_score = evaluate(optimized_program)

print(f"Baseline Score: {baseline_score}")
print(f"Optimized Score: {optimized_score}")

# 8. 保存优化后的程序
optimized_program.save("optimized_bbh_program", save_program=True)


# 9. 保存语言模型历史记录到 JSON 文件
def save_lm_history(filename):
    """将全局LM历史记录保存到 JSON 文件，重点关注 usage 和 cost 信息"""
    if not os.path.exists("lm_history"):
        os.makedirs("lm_history")

    # 使用 dspy 的全局历史记录
    history = GLOBAL_HISTORY

    if history:
        # 创建可序列化的历史记录列表，专注于 usage 和 cost
        serializable_history = []
        for entry in history:
            # 处理 usage 字段：确保它是基本数据类型
            usage_dict = {}
            if "usage" in entry and entry["usage"]:
                usage = entry["usage"]
                if hasattr(usage, "items"):  # 如果是类字典对象
                    for k, v in usage.items():
                        usage_dict[k] = str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
                else:
                    usage_dict = {"info": str(usage)}

            # 创建安全的条目字典
            simplified_entry = {
                "timestamp": str(entry.get("timestamp", "")),
                "model": str(entry.get("model", "")),
                "response_model": str(entry.get("response_model", "")),
                "uuid": str(entry.get("uuid", "")),
                "usage": usage_dict,
                "cost": float(entry.get("cost", 0)) if entry.get("cost") is not None else None,
                "prompt_brief": str(entry.get("prompt", ""))[:100] + "..." if len(
                    str(entry.get("prompt", ""))) > 100 else str(entry.get("prompt", "")),
            }
            serializable_history.append(simplified_entry)

        filepath = os.path.join("lm_history", f"{filename}.json")

        # 使用更安全的方式进行序列化
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_history, f, ensure_ascii=False, indent=2)
            print(f"保存语言模型历史记录到: {filepath}, 包含 {len(serializable_history)} 条记录")
        except TypeError as e:
            print(f"错误: 序列化失败 - {e}")
            # 尝试更简单的方法 - 进一步简化记录
            basic_history = []
            for entry in serializable_history:
                basic_entry = {k: str(v) for k, v in entry.items()}
                basic_history.append(basic_entry)

            # 使用最基本的字符串表示保存
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(basic_history, f, ensure_ascii=False, indent=2)
            print(f"已使用基本字符串表示保存历史记录")
    else:
        print(f"警告: 全局历史记录为空，没有历史记录可保存")


# 保存全局历史记录，专注于 usage 和 cost 信息
save_lm_history("dspy_global_history")
