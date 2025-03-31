import dspy
import json
import os
from bbh import read_jsonl_bbh, calculate_score_bbh  # 从 bbh.py 导入相关功能
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
def save_lm_history(lm, filename):
    """将语言模型的历史记录保存到 JSON 文件"""
    if not os.path.exists("lm_history"):
        os.makedirs("lm_history")
    
    history = lm.history
    if history:
        filepath = os.path.join("lm_history", f"{filename}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"保存语言模型历史记录到: {filepath}")
    else:
        print(f"警告: {filename} 没有历史记录可保存")

# 保存两个语言模型的历史记录
save_lm_history(prompt_lm, "prompt_lm_history")
save_lm_history(task_lm, "task_lm_history")

# 如果希望在实验中途也保存历史记录，可以在关键步骤后添加保存操作
# 例如，在优化前后分别保存一次:
# save_lm_history(task_lm, "task_lm_before_optimization")
# save_lm_history(task_lm, "task_lm_after_optimization")
