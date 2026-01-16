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


def truncate_interest(interest: str, max_length: int = 7) -> str:
    """
    截断interest文本，超过max_length长度的中间部分用省略号替换
    
    Args:
        interest: 要截断的interest文本
        max_length: 最大文本长度
        
    Returns:
        截断后的interest文本
    """
    if len(interest) <= max_length:
        return interest
    
    # 保留前3个和后3个字符，中间用中文省略号替换
    return interest[:3] + "…" + interest[-3:]


def calculate_paper_score(paper: ArxivPaper, interests: List[str]) -> float:
    from .llm import get_llm
    llm = get_llm()

    # 1. 定义 Few-Shot 例子
    # 这里的例子要涵盖：极其相关、中等相关、完全不相关三种情况
    few_shot_context = """
Example 1: High Relevance
- User Interests: ["AI Agents", "Software Testing"]
- Title: "DeepRegression: A Multi-Agent Reinforcement Learning Framework for Autonomous Flaky Test Identification"
- Abstract: "Non-deterministic test failures, or flaky tests, represent a significant bottleneck in modern CI/CD pipelines. We introduce DeepRegression, a framework that deploys a swarm of LLM-based agents to autonomously explore execution traces and commit histories. Unlike static analyzers, our agents utilize Reinforcement Learning (RL) to develop search policies that prioritize high-entropy log segments. We evaluated DeepRegression on a dataset of 1,000 Open-source projects, demonstrating a 25% improvement in root cause localization. The system features a self-correcting feedback loop where agents refine their prompts based on the success of proposed patches."
- Internal Logic:
AI Agents: Domain(10), Problem(10), Method(10) -> (3.0 + 3.0 + 4.0) * 10 = 100
Software Testing: Domain(10), Problem(10), Method(9) -> (3.0 + 3.0 + 3.6) * 10 = 96
- Output: {"AI Agents": 100, "Software Testing": 96}

Example 2: Methodological Synergy
- User Interests: ["AI Agents", "Software Testing"]
- Title: "Adaptive Resource Allocation in Cloud Data Centers via Decentralized Autonomous Agents"
- Abstract: "Efficient resource management is critical for minimizing latency in cloud environments. This paper proposes a decentralized architecture where individual compute nodes are managed by autonomous agents. These agents employ a hierarchical planning strategy to negotiate task offloading under strict SLA constraints. We utilize a transformer-based world model to predict future workload spikes. While the application is infrastructure management, the multi-agent negotiation protocols and predictive modeling of system uncertainty offer significant parallels to resolving concurrency issues in complex software build environments."
- Internal Logic:AI Agents: 
Domain(7), Problem(4), Method(9) -> (2.1 + 1.2 + 3.6) * 10 = 69
Software Testing: Domain(3), Problem(2), Method(4) -> (0.9 + 0.6 + 1.6) * 10 = 31
- Output: {"AI Agents": 69, "Software Testing": 31}

Example 3: Marginal Relevance
- User Interests: ["AI Agents", "Software Testing"]
- Title: "A Qualitative Study on the Impact of Generative AI Chatbots in Undergraduate Software Engineering Education"
- Abstract: "As Generative AI becomes ubiquitous, its role in education must be examined. We conducted a semester-long study to observe how undergraduate students use AI-powered chatbots (referred to as educational agents) during their introductory software testing course. Through surveys and interview data, we analyzed whether these tools helped students write better unit tests or if they encouraged academic dishonesty. Our findings suggest that while these 'agents' can provide quick templates, they often hallucinate incorrect testing syntax and do not significantly improve the students' conceptual understanding of code coverage."
- Internal Logic:AI Agents: Domain(2), Problem(1), Method(1) -> (0.6 + 0.3 + 0.4) * 10 = 13
Software Testing: Domain(3), Problem(2), Method(1) -> (0.9 + 0.6 + 0.4) * 10 = 19
- Output: {"AI Agents": 13, "Software Testing": 19}
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
        
        ### [Constraints]
        - Capture micro-details from the abstract for scoring, rather than relying solely on high-level title keywords.
        - Strictly forbid "binary" scores (such as 0 or 100) unless the paper is entirely irrelevant.
        - Aim for distinguishable, granular scores (e.g., 74, 58, 82) to reflect subtle differences in relevance.

        ### Current Task:
        User Interests: {target_interests}
        Paper Title: {paper.title}
        Paper Abstract: {paper.summary[:1200]}

        ### Requirement:
        - Return ONLY the JSON object.
        - Scores must be integers between 0 and 100.
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
            
            # 收集所有分数>=80的interest，并截断长文本
            paper.high_score_interests = []
            if scores_dict:
                for interest, score in scores_dict.items():
                    if score >= 80:
                        truncated_interest = truncate_interest(interest)
                        paper.high_score_interests.append(truncated_interest)
                
                # 设置总分数为最高分
                paper.score = float(max(scores_dict.values()))
            
            return paper.score
            
    except Exception as e:
        logger.error(f"Few-shot scoring failed: {e}")
        paper.score = 0.0
        return paper.score
    
    paper.score = 0.0
    return paper.score