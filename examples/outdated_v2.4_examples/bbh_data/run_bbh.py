import datetime
import json
import os

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


# 9. 保存语言模型历史记录和实验结果
def safe_serializable(obj):
    """将对象转换为安全可序列化的形式"""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, list):
        return [safe_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): safe_serializable(v) for k, v in obj.items() if k != 'response'}
    else:
        return str(obj)


def extract_usage_info(history_entry):
    """从历史记录条目中提取使用信息"""
    usage_info = {}
    if "usage" in history_entry and history_entry["usage"]:
        usage = history_entry["usage"]
        # 处理不同类型的usage对象
        if hasattr(usage, "get"):
            usage_info = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
        else:
            # 回退到字符串表示
            usage_info = {"info": str(usage)}

    return usage_info


def save_lm_history(lm, filename, output_dir):
    """保存单个语言模型的历史记录"""
    if not hasattr(lm, 'history') or not lm.history:
        print(f"警告: {filename} 没有历史记录可保存")
        return False

    try:
        # 提取关键信息，特别是usage和cost
        simplified_history = []
        for entry in lm.history:
            # 提取基本信息
            entry_info = {
                "timestamp": str(entry.get("timestamp", "")),
                "model": str(entry.get("model", "")),
                "uuid": str(entry.get("uuid", "")),
                "usage": extract_usage_info(entry),
                "cost": entry.get("cost")
            }

            # 添加少量上下文信息
            if "prompt" in entry:
                prompt_str = str(entry["prompt"])
                entry_info["prompt"] = prompt_str

            # 添加输出的简要信息
            if "outputs" in entry and entry["outputs"]:
                outputs = entry["outputs"]
                if isinstance(outputs, list) and outputs:
                    output_str = str(outputs[0])
                    entry_info["outputs"] = output_str

            simplified_history.append(entry_info)

        # 保存到文件
        filepath = os.path.join(output_dir, f"{filename}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(simplified_history, f, ensure_ascii=False, indent=2)

        print(f"成功保存 {len(simplified_history)} 条历史记录到: {filepath}")
        return True
    except Exception as e:
        print(f"保存 {filename} 时出错: {e}")

        # 尝试更简单的方法 - 只保存字符串表示
        try:
            basic_history = [{"entry": str(entry)} for entry in lm.history]
            filepath = os.path.join(output_dir, f"{filename}_basic.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(basic_history, f, ensure_ascii=False, indent=2)
            print(f"已使用基本字符串表示保存历史记录到: {filepath}")
            return True
        except Exception as e2:
            print(f"基本保存也失败: {e2}")
            return False


def save_experiment_results():
    """保存实验结果，包括各个语言模型的历史记录和评分"""
    # 创建输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("experiment_results", f"bbh_experiment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 保存prompt_lm和task_lm的历史记录
    prompt_saved = save_lm_history(prompt_lm, "prompt_lm_history", output_dir)
    task_saved = save_lm_history(task_lm, "task_lm_history", output_dir)

    # 保存实验结果摘要
    results = {
        "timestamp": timestamp,
        "baseline_score": float(baseline_score),
        "optimized_score": float(optimized_score),
        "prompt_lm_model": str(prompt_lm.model),
        "task_lm_model": str(task_lm.model),
        "history_saved": {
            "prompt_lm": prompt_saved,
            "task_lm": task_saved
        }
    }

    # 更安全地检查和提取优化后的指令和示例
    try:
        # 检查MIPROv2优化后的程序结构
        optimized_info = {}

        # 检查是否有指令（多种可能的位置）
        if hasattr(optimized_program, "generate_answer"):
            # 直接获取生成答案组件
            cot = optimized_program.generate_answer

            # 检查不同可能的属性路径
            if hasattr(cot, "predictor"):
                predictor = cot.predictor
                if hasattr(predictor, "instruction"):
                    optimized_info["instruction"] = predictor.instruction
                if hasattr(predictor, "demos"):
                    optimized_info["has_demos"] = True
                    optimized_info["demo_count"] = len(predictor.demos)

            # 也可能直接在ChainOfThought上
            elif hasattr(cot, "instruction"):
                optimized_info["instruction"] = cot.instruction

            # 可能在模块属性中
            elif hasattr(cot, "__dict__"):
                for key, value in cot.__dict__.items():
                    if "instruction" in key.lower() and isinstance(value, str):
                        optimized_info[f"instruction_in_{key}"] = value
                    if "demo" in key.lower():
                        optimized_info[f"demos_in_{key}"] = str(type(value))

        # 如果能获取到teleprompter信息
        if hasattr(teleprompter, "best_instruction"):
            optimized_info["best_instruction"] = teleprompter.best_instruction

        # 保存找到的所有优化信息
        if optimized_info:
            results["optimized_program_info"] = optimized_info
    except Exception as e:
        # 如果提取过程中出错，记录错误但继续
        results["optimized_program_info_error"] = str(e)

    # 保存结果摘要
    results_path = os.path.join(output_dir, "experiment_summary.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"已保存实验结果摘要到: {results_path}")
    return output_dir


# 执行保存
output_dir = save_experiment_results()
print(f"实验数据已保存到: {output_dir}")
