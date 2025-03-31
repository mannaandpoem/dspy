import dspy
from dspy.datasets import HotPotQA
from dspy.evaluate import Evaluate
from dspy.teleprompt import MIPROv2

# 1. 配置语言模型和检索模型
api_key = ...  # 替换为你的 API Key
base_url = ...  # 替换为你的 base_url（如果有的话）

# 实例化两个不同的语言模型
prompt_lm = dspy.LM('gpt-4o', api_key=api_key, base_url=base_url)  # 用于生成提示的模型
task_lm = dspy.LM('gpt-4o-mini', api_key=api_key, base_url=base_url)  # 用于执行任务的模型

# 配置默认的语言模型和检索模型（这里仍使用 task_lm 作为全局默认）
colbertv2 = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.configure(lm=task_lm, rm=colbertv2)  # 默认使用 task_lm

# 2. 定义一个简单的问答模块
class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate_answer(context=context, question=question)

# 3. 加载数据
dataset = HotPotQA(train_seed=1, train_size=100, eval_seed=2023)
trainset = [x.with_inputs('question') for x in dataset.train[:1]]
valset = [x.with_inputs('question') for x in dataset.dev[:1]]

# 4. 定义评估指标
def exact_match_metric(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

evaluate = Evaluate(devset=valset, metric=exact_match_metric, num_threads=4, display_progress=True)

# 5. 初始化程序
program = SimpleQA()

# 6. 运行 MIPROv2 优化
teleprompter = MIPROv2(
    prompt_model=prompt_lm,  # 使用 gpt-4o 作为 prompt_model
    task_model=task_lm,      # 使用 gpt-4o-mini 作为 task_model
    metric=exact_match_metric,
    num_candidates=2,        # 生成2个候选指令和few-shot示例
    init_temperature=1.0,
    verbose=True
)

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
    num_trials=2,            # 运行2个优化批次
    max_bootstrapped_demos=1,
    max_labeled_demos=1,
    minibatch=True,          # 启用 minibatch，使用默认值
)

# 7. 评估优化后的程序
baseline_score = evaluate(program)
optimized_score = evaluate(optimized_program)

print(f"Baseline Score: {baseline_score}")
print(f"Optimized Score: {optimized_score}")

# 8. 保存优化后的程序
optimized_program.save("optimized_qa_program.dspy", save_program=True)