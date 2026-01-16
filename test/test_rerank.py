import json
from src.rerank import rerank_paper  # 确保路径正确

# 1. 模拟论文对象类
class MockPaper:
    def __init__(self, arxiv_id, title, summary):
        self.arxiv_id = arxiv_id
        self.title = title
        self.summary = summary
        self.score = 0

# 2. 准备测试用例
def run_semantic_test():
    # 兴趣目标：LLM 算法与推理优化
    target = "Large Language Models Reasoning and Inference Optimization"
    # target = "Applications of Large Language Models"

    test_papers = [
        # --- 1. 核心目标 (应得高分: 90-100) ---
        MockPaper(
            "2401.STEP",
            "STEP: Step-level Trace Evaluation and Pruning",
            "We propose a novel pruning framework that evaluates reasoning steps using hidden states and dynamically prunes unpromising traces during generation to reduce end-to-end latency in LLMs."
        ),

        # --- 2. 应用层陷阱 (应得低分: < 20) ---
        # 虽然包含 LLM，但核心是行政/翻译工具使用，无算法创新
        MockPaper(
            "TRAP.APP",
            "Improving Internal Corporate Newsletters with LLM-based Translation",
            "We demonstrate a workflow for using ChatGPT to translate internal corporate memos into five different languages to improve employee engagement in multinational firms."
        ),

        # --- 3. 术语混淆陷阱 (应得低分: < 10) ---
        # 利用“Attention”在生物领域的含义，测试模型是否只看关键词
        MockPaper(
            "TRAP.BIO",
            "Attention Mechanisms in Protein Folding Sequences",
            "In this biological study, we analyze the attention patterns of amino acid chains during protein synthesis. We identify how specific sequences attract molecular binders."
        ),

        # --- 4. 软科学/伦理陷阱 (应得低分: < 15) ---
        # 讨论社会影响而非硬核技术
        MockPaper(
            "TRAP.SOC",
            "The Sociological Impact of Generative AI on Remote Work Culture",
            "Through a series of interviews, we explore how the rise of LLMs has changed the way remote workers perceive their job security and daily social interactions."
        ),

        # --- 5. 纯硬件/设施陷阱 (应得低分: 0-10) ---
        # 虽然提到 LLM GPU，但属于土木/暖通工程
        MockPaper(
            "2401.COOL",
            "Liquid Cooling Systems for GPU Clusters",
            "Optimizing liquid cooling systems for data centers hosting massive H100 GPU clusters used for Large Language Models (LLM) inference to prevent thermal throttling."
        ),

        # --- 6. 法律/合规陷阱 (应得低分: 10-20) ---
        # 法律分析而非算法改进
        MockPaper(
            "2401.LEGAL",
            "Copyright Infringement in AI Training",
            "A legal analysis of copyright infringement liability regarding training data used in Large Language Models (LLM) and the implications for digital intellectual property law."
        ),

        # --- 7. 硬核但无关陷阱 (应得低分: 0) ---
        # 技术深度很高，但领域完全错位（数据库索引）
        MockPaper(
            "TRAP.DB",
            "B-Tree Indexing Optimization for Real-time SQL Queries",
            "We propose a novel dynamic B-Tree rebalancing algorithm that reduces disk I/O latency by 30% for high-concurrency SQL databases in large-scale distributed systems."
        ),

        # --- 8. 机器人/物理动作陷阱 (应得低分: < 20) ---
        # 测试对“Action-based”一词的理解（物理动作 vs 逻辑推理）
        MockPaper(
            "TRAP.ROBOT",
            "Trajectory Planning for Quadrupedal Robots in Rugged Terrain",
            "This paper presents a reinforcement learning approach for real-time action planning in four-legged robots to maintain stability while traversing uneven rocky surfaces."
        )
    ]

    print(f"🔍 Testing Semantic Discrimination for: '{target}'\n")

    # 3. 调用你的重排序函数
    # 注意：这里会加载本地 Llama 模型，第一次运行可能较慢
    results = rerank_paper(test_papers, target)

    # 4. 结果分析
    print("\n--- Test Results ---")
    for i, p in enumerate(results):
        status = "✅ PASS" if (p.arxiv_id == "2401.STEP" and i == 0) else "❌ FAIL"
        # 理想情况是 STEP 拿第一，且分数远高于其他两篇
        print(f"Rank {i+1}: [{p.score}] {p.title}")
        if p.arxiv_id == "2401.STEP":
            target_score = p.score
        else:
            distractor_score = p.score

    print("\n--- Analysis ---")
    if target_score > distractor_score + 30:
        print("🎯 Great! The model distinguishes between 'Core Tech' and 'Peripheral Keywords'.")
    else:
        print("⚠️ Warning: The score gap is too small. You might need to refine the Prompt.")

if __name__ == "__main__":
    run_semantic_test()