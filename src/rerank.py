import logging
from typing import List, Tuple
from .paper import ArxivPaper
from .llm import LLM, set_global_llm
import time

logger = logging.getLogger(__name__)

def rerank_paper(papers: List[ArxivPaper], retriever_target: str, model: str = "meta-llama/Llama-3.2-3B-Instruct", model_url: str = "llama-3.2-3b-instruct.Q4_K_M.gguf") -> List[ArxivPaper]:
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


def calculate_paper_score(paper: ArxivPaper, interests: List[str]) -> float:
    """
    计算论文与感兴趣方向的相关性分数
    
    Args:
        paper: 论文对象
        interests: 感兴趣的方向列表
    
    Returns:
        最高相关性分数（0-100）
    """
    from llm import get_llm
    llm = get_llm()
    
    # 少样本提示示例
    few_shot_examples = """
    Example 1:
    Interest: Machine Learning
    Paper Title: Attention Is All You Need
    Paper Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...
    Score: 95
    
    Example 2:
    Interest: Computer Vision
    Paper Title: Attention Is All You Need
    Paper Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...
    Score: 70
    
    Example 3:
    Interest: Quantum Computing
    Paper Title: Attention Is All You Need
    Paper Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...
    Score: 20
    """
    
    # 为每个感兴趣的方向单独打分
    scores = []
    for interest in interests:
        prompt = f"""
        You are an expert in academic paper analysis. Your task is to score the relevance of a paper to a specific research interest on a scale of 0 to 100, where 100 is extremely relevant and 0 is completely irrelevant.
        
        {few_shot_examples}
        
        Interest: {interest}
        Paper Title: {paper.title}
        Paper Abstract: {paper.summary}
        Score:
        """
        
        try:
            response = llm.generate([
                {"role": "system", "content": "You are an expert in academic paper analysis."},
                {"role": "user", "content": prompt}
            ])
            
            # 解析分数
            score = float(response.strip())
            scores.append(score)
            logger.debug(f"  Interest: {interest}, Score: {score}")
        except Exception as e:
            logger.error(f"Failed to calculate score for interest '{interest}': {e}")
            scores.append(0)
    
    # 返回最高分
    if scores:
        max_score = max(scores)
        logger.debug(f"  Maximum score: {max_score}")
        return max_score
    else:
        return 0
