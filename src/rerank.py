from typing import List, Tuple
from .paper import ArxivPaper
from .llm import LLM, set_global_llm
import time
from loguru import logger
import json

def rerank_paper(papers: List[ArxivPaper], retriever_target: str, model: str = "bartowski/Llama-3.2-3B-Instruct-GGUF", model_url: str = "Llama-3.2-3B-Instruct-Q4_K_M.gguf") -> List[ArxivPaper]:
    """
    使用Llama-3.2-3B-Instruct模型对论文进行语义排序
    
    Args:
        papers: 论文列表
        retriever_target: 感兴趣的方向，用换行符分隔
    
    Returns:
        排序后的论文列表
    """
    # 1. 解析感兴趣的方向
    interests = [interest.strip() for interest in retriever_target.split('\n') if interest.strip()]
    if not interests:
        logger.warning("No valid interests provided, returning papers in original order")
        return papers
    
    logger.info(f"Reranking {len(papers)} papers based on interests: {interests}")
    
    # 2. 初始化Llama-3.2-3B-Instruct模型
    set_global_llm(model=model, model_path=model_url)
    
    # 3. 为每个论文打分
    scored_papers = []
    for i, paper in enumerate(papers):
        logger.info(f"Scoring paper {i+1}/{len(papers)}: {paper.title[:50]}...")
        try:
            score = calculate_paper_score(paper, interests)
            paper.score = score
            scored_papers.append((score, paper))
            logger.info(f"  Score: {score}")
        except Exception as e:
            logger.error(f"Failed to score paper {paper.arxiv_id}: {e}")
            paper.score = 0
            scored_papers.append((0, paper))
        
        # 添加延迟，避免模型过载
        time.sleep(0.5)
    
    # 4. 按分数降序排序
    sorted_papers = [paper for _, paper in sorted(scored_papers, key=lambda x: x[0], reverse=True)]
    
    logger.info(f"Reranking completed. Top 5 papers:")
    for i, paper in enumerate(sorted_papers[:5]):
        logger.info(f"  {i+1}. {paper.title[:50]}... (Score: {paper.score})")
    
    return sorted_papers


def truncate_interest(interest: str, max_length: int = 40) -> str:
    """
    截断interest文本，超过max_length长度的末尾用省略号替换
    
    Args:
        interest: 要截断的interest文本
        max_length: 最大文本长度（默认40个字符）
        
    Returns:
        截断后的interest文本
    """
    if len(interest) <= max_length:
        return interest
    
    # 固定长度40个字符，超出部分在后面使用...隐藏
    return interest[:max_length - 3] + "..."


def calculate_paper_score(paper: ArxivPaper, interests: List[str]) -> float:
    from .llm import get_llm
    llm = get_llm()

    # 1. 定义 Few-Shot 例子
    # 这里的例子要涵盖：极其相关、中等相关、完全不相关三种情况
    few_shot_context = """
    ### Examples:
    User Interests: ["LLM", "Software Testing"]

    Example 1 (Highly Relevant):
    Title: "Unit Test Generation using Large Language Models"
    Abstract: "This paper investigates the effectiveness of using Large Language Models (LLMs) like GPT-4 to automate unit test generation for Java projects. We evaluate the syntactic correctness and coverage of the generated tests compared to traditional search-based software testing (SBST) techniques."
    Output: {"LLM": 95, "Software Testing": 98}

    Example 2 (Partially Relevant):
    Title: "Auto-GPT: An Autonomous GPT-4 Experiment for Business Automation"
    Abstract: "We present an open-source experiment to make GPT-4 fully autonomous. By chaining LLM "thoughts", the system can independently achieve goals like market research and code debugging. We analyze the reliability and safety challenges in these autonomous loops."
    Output: {"LLM": 92, "Software Testing": 40}

    Example 3 (Irrelevant):
    Title: "Quantum Approximate Optimization Algorithms for Graph Coloring"
    Abstract: "We propose a hybrid quantum-classical algorithm for the graph coloring problem. By utilizing QAOA on a 50-qubit processor, we demonstrate a speedup in finding optimal colorings for sparse graphs."
    Output: {"LLM": 5, "Software Testing": 0}
    """

    # 2. 构造当前的任务 Prompt
    # 将你的 Interests 列表转为 JSON 字符串
    target_interests = json.dumps(interests)
    
    prompt = f"""
        ### [Detailed Interest Analysis]
        Please evaluate the paper based on these three specific dimensions (0-10 points each):
        
        1. Domain Overlap (0-10):
           - Assess whether the research scope falls within the sub-fields defined in [RETRIEVER_TARGET].
           - 0: Completely irrelevant; 10: Core domain is highly consistent.
        
        2. Problem Alignment (0-10):
           - Does the specific problem addressed (e.g., efficiency, reliability, novelty, robustness) align with the user's primary concerns?
           - Consider if the research motivation addresses practical pain points in the user's research or engineering workflows.
        
        3. Methodological Synergy (0-10):
           - Does the technical approach (e.g., Reinforcement Learning, Chain-of-Thought, Edge Detection, Model Compression) match the user's technical stack?
           - Even if the domain slightly diverges, check if the implementation provides direct reference or inspirational value.
        
        ### [Scoring Logic]
        Final Score Calculation Formula:
        $$Total Score = (Domain * 0.3 + Problem * 0.3 + Method * 0.4) * 10$$

        ### [Example]
        {few_shot_context}

        ### [Current Task]
        Current User Interests: {target_interests}
        Paper Title: {paper.title}
        Paper Abstract: {paper.summary[:1200]}

        ### [Requirement]
        - Return ONLY the JSON object.
        - The keys in the JSON must EXACTLY match the provided "Current User Interests" list.
        - Include EVERY interest from the list, even if the score is low. Do not add any extra categories.
        - Scores must be integers between 0 and 100.
        - Granularity Trigger: Avoid scores that are multiples of 5 (e.g., 70, 75, 80) to ensure higher ranking granularity
        - No explanation.

        Output:"""

    try:
        response = llm.generate([
            {"role": "system", "content": "You are a research assistant that evaluates paper relevance in JSON format."},
            {"role": "user", "content": prompt}
        ])

        # 3. 稳健的 JSON 提取
        import re
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            scores_dict = json.loads(match.group())
            
            # 更新paper对象的interest_scores属性
            paper.interest_scores = scores_dict

            print(scores_dict)
            
            # 收集所有分数>=80的interest，存储完整文本
            paper.high_score_interests = []
            if scores_dict:
                for interest, score in scores_dict.items():
                    if score >= 80:
                        paper.high_score_interests.append(interest)
                
                # 设置总分数为最高分
                paper.score = float(max(scores_dict.values()))
            
            return paper.score
            
    except Exception as e:
        logger.error(f"Few-shot scoring failed: {e}")
        paper.score = 0.0
        return paper.score
    
    paper.score = 0.0
    return paper.score