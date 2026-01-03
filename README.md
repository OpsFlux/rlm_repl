# 递归语言模型（RLMs）实现

本仓库提供一个“递归语言模型（Recursive Language Models, RLMs）”的概念验证实现，基于 Alex L. Zhang、Tim Kraska、Omar Khattab 的论文《Recursive Language Models》（https://arxiv.org/pdf/2512.24601）。实现思路也参考了 https://github.com/alexzhang13/rlm-minimal 中的最小示例代码。

## 目录
- [概览](#overview)
- [核心组件](#core-components)
- [关键特性](#key-features)
- [安装](#installation)
- [配置](#configuration)
- [用法](#usage)
- [性能特征](#performance-characteristics)
- [参考资料](#references)

<a id="overview"></a>
## 概览

该实现提供了一个递归语言模型（RLM）系统：通过“推理时扩展（inference-time scaling）”来让 LLM 能处理任意长度的提示（prompt）。核心做法是把 prompt（尤其是超长上下文）当作外部环境的一部分，让模型在推理过程中以编程方式去检查、分解上下文，并对 prompt 的片段进行递归自调用。

<a id="core-components"></a>
## 核心组件

### 1. RLM 基类（`rlm/rlm.py`）
- 定义 RLM 接口的抽象基类
- 主要方法：`completion()`、`cost_summary()`、`reset()`

### 2. 带 REPL 的 RLM（`rlm/rlm_repl.py`）
- 基于 REPL 环境的主要 RLM 实现
- 上下文存放在 REPL 外部状态中，不直接随每次调用传给模型
- 同时支持 root_model 与 sub_model（递归子调用模型）
- 分别跟踪 root 与 sub-LLM 的调用成本
- 带终止条件的迭代交互循环

### 3. REPL 环境（`rlm/repl.py`）
- Python 执行沙箱
- 用于递归调用的 `llm_query(prompt)` 函数
- 跨迭代的状态持久化
- 输出捕获（stdout/stderr）
- 中间变量与结果管理

### 4. 工具与辅助模块（`rlm/utils/`）
- `llm.py`：LLM 客户端封装与成本统计
- `prompts.py`：论文附录 D 的系统提示词（system prompts）
- `tracing.py`：更细粒度的日志与追踪系统

<a id="key-features"></a>
## 关键特性

### 上下文管理
- 上下文作为 REPL 环境中的外部变量存储
- 向 LLM 提供关于上下文大小与结构的元信息
- 让系统可处理超过模型上下文窗口限制的输入

### 递归子调用
- 通过 `llm_query` 触发递归式的 LLM 调用
- root 与 sub-LLM 成本分开统计
- 支持对大上下文进行分块（chunking）与分步处理

### 代码执行
- 在 REPL 环境中执行 Python 代码
- 允许以编程方式检查、筛选与分解上下文
- 多轮迭代保持状态

### 最终答案处理
- 使用 `FINAL()` 与 `FINAL_VAR()` 作为终止响应的机制
- 对终止条件进行严格检查

### 成本统计
- 分别统计 root 与 sub-LLM 的成本
- 统计输入/输出 token
- 统计调用次数，便于分析

### 可配置的 API Endpoint
- 支持通过环境变量 `RLM_API_URL` 指定接口地址
- 通过 `/models` 自动探测可用模型，并在未指定时选择第一个可用模型
- 默认回退到 `http://localhost:8080/v1`

### 增强的配置项
- `max_iterations`：root LLM 最大迭代次数（默认：20）
- `max_output_length`：REPL 输出截断前的最大长度（默认：500,000 字符）
- 达到最大迭代次数时返回 `None`，而不是强行生成答案，以贴近论文描述的自然收敛过程
- 提升输出长度上限，降低长输出任务的截断影响

<a id="installation"></a>
## 安装

1. 克隆仓库：
```bash
git clone https://github.com/fullstackwebdev/rlm_repl
cd rlm_repl
```

2. 安装依赖（如有）：
```bash
# 本实现使用标准 Python 库
# 无需额外依赖
```

3. 准备本地 LLM Server（例如 llama.cpp server、vLLM 等）。

<a id="configuration"></a>
## 配置

该实现支持可配置的 API endpoint：

1. 通过环境变量 `RLM_API_URL` 指向你的 LLM 服务：
```bash
export RLM_API_URL="http://your-llm-server:port/v1"
```

2. 未设置时，默认使用 `http://localhost:8080/v1`。

3. 系统会从 `/models` endpoint 自动探测可用模型；若未指定模型名，则使用第一个可用模型。

<a id="usage"></a>
## 用法

### 基本用法
```python
from rlm.rlm_repl import RLM_REPL

# 创建 RLM 实例
rlm = RLM_REPL(
    model="auto",  # 自动选择第一个可用模型
    recursive_model="auto",  # 自动选择第一个可用模型
    max_iterations=10
)

# 处理长上下文
result = rlm.completion(
    context="很长的上下文……",
    query="这个问题的答案是什么？"
)

# 查看成本统计
costs = rlm.cost_summary()
print(f"总成本: ${costs['total_cost']:.4f}")
```

### 指定模型
```python
from rlm.rlm_repl import RLM_REPL

# 使用指定模型创建 RLM 实例
rlm = RLM_REPL(
    model="Qwen3-Coder-REAP-25B-A3B.Q5_K_M.gguf",
    recursive_model="Qwen3-Coder-REAP-25B-A3B.Q5_K_M.gguf",
    max_iterations=10,
    max_output_length=500000  # 截断前的字符数上限
)

# 处理长上下文
result = rlm.completion(
    context="很长的上下文……",
    query="这个问题的答案是什么？"
)

# 注意：如果达到 max_iterations 仍未得到最终答案，result 可能为 None
if result is None:
    print("RLM 达到最大迭代次数，但未找到最终答案")
else:
    print(f"结果: {result}")
```

<a id="performance-characteristics"></a>
## 性能特征

- 可处理显著超过模型上下文窗口限制的超长上下文
- 质量与基础 LLM 或常见长上下文脚手架相比相当或更好
- 单次查询成本与替代方案相比相当或更低
- 随着上下文长度与任务复杂度增加，仍能保持较强表现

<a id="references"></a>
## 参考资料

- Zhang, A. L., Kraska, T., & Khattab, O. (2025). Recursive Language Models. arXiv preprint arXiv:2512.24601.
- https://arxiv.org/pdf/2512.24601
- https://github.com/alexzhang13/rlm-minimal
