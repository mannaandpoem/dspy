import dspy
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
        self.total_cost = 0.0  # 用于累计成本

    def forward(self, question):
        prediction = self.generate_answer(question=question)
        # 从语言模型的最后一次调用中获取成本（如果支持）
        cost = dspy.settings.lm.last_call_cost if hasattr(dspy.settings.lm, 'last_call_cost') else 0.0
        self.total_cost += cost
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


# 自定义评估函数以记录成本
# evaluate = Evaluate(devset=valset, metric=bbh_exact_match_metric, num_threads=4, display_progress=True)
def evaluate_with_cost(program, devset, metric):
    evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
    score = evaluator(program)
    total_cost = program.total_cost  # 从程序实例中获取累计成本
    return score, total_cost


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
# baseline_score = evaluate(program)
# optimized_score = evaluate(optimized_program)
# 7. 评估优化前后的程序并记录成本
baseline_score, baseline_cost = evaluate_with_cost(program, valset, bbh_exact_match_metric)
optimized_score, optimized_cost = evaluate_with_cost(optimized_program, valset, bbh_exact_match_metric)

print(f"Baseline Score: {baseline_score}")
print(f"Optimized Score: {optimized_score}")

# 8. 保存优化后的程序
optimized_program.save("optimized_bbh_program", save_program=True)
