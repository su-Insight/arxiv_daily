<p align="center">
  <a href="" rel="noopener">
 <img width="125" height="125" alt="image" src="https://github.com/user-attachments/assets/a1dfc1c6-ab3b-4ca1-a9df-036d7a97bb80" /></a>
</p>
<h3 align="center">ArXiv Radar</h3>

<div align="center">

  ![Status](https://img.shields.io/badge/status-active-success.svg)
  ![Stars](https://img.shields.io/github/stars/su-Insight/arxiv_radar?style=flat)
  ![Issues](https://img.shields.io/github/issues/su-Insight/arxiv_radar)
  ![Pull Requests](https://img.shields.io/github/issues-pr/su-Insight/arxiv_radar)
  ![License](https://img.shields.io/github/license/su-Insight/arxiv_radar)

</div>

<p align="center"> 🚀 每日根据您的科研兴趣，利用本地 LLM 智能检索、语义重排并推送 ArXiv 最新论文。
    <br> 
</p>

> [!IMPORTANT]
> 项目处于活跃开发阶段。请关注本仓库，并在上游更新时及时同步（Sync）您的 Fork 仓库，以享受最新的语义判别逻辑和功能修复。

## 🧐 关于项目 <a name = "about"></a>

> 想要紧跟科研前沿？只需 Fork (并 Star⭐) 本仓库即可开启您的专属论文雷达！😊

**ArXiv Radar** 是一款能够精准捕捉您科研兴趣的自动化工具。它不同于简单的关键词检索，而是利用本地轻量级大语言模型（Llama3.2-3B）对每日论文进行**深度语义判别**。系统会分析论文的核心贡献，区分“硬核技术创新”与“普通应用落地”，确保推送到您邮箱的内容都是真正有价值的研究。

依托 **Github Action**，它可以实现 **零成本**、**免安装**、**全自动** 的每日学术快报服务。

## ✨ 特性

- **完全免费**: 所有计算任务均在 Github Action 赠送的配额内完成，无需自有服务器。
- **语义重排**: 利用 LLM 的少样本、思维链（CoT）能力对论文进行相关性评分，有效过滤信息噪音。
- **AI 生成 TLDR**: 为每篇论文生成精简的总结（支持中文），助您秒级筛选核心信息。
- **富文本邮件**: 集成 PDF 直达链接、代码仓库地址、作者所属机构、高分主题样式，支持邮件内交互过滤。
- **本地推理能力**: 默认下载并运行本地模型，保护您的科研兴趣隐私，无需支付 API 费用。
- **可选 API 支持**: 如果您追求更快的生成速度或更高的质量，也支持云端 API 接入。

## 📷 效果展示

![screenshot](https://via.placeholder.com/800x600?text=ArXiv+Radar+Email+Screenshot)

## 🚀 使用指南

### 快速开始

1. **Fork 本仓库** (别忘了点个 Star 😉)。

2. **设置 GitHub Secrets**：
   进入仓库 `Settings -> Secrets and variables -> Actions`，添加以下 `Repository secrets`：

| 变量名 | 必填 | 类型 | 描述 | 示例 |
| :--- | :---: | :--- | :--- | :--- |
| ARXIV_QUERY | ✅ | str | 检索的 ArXiv 分类，用 `+` 连接，[明细详见](https://arxiv.org/category_taxonomy)。 | cs.AI+cs.CV+cs.LG |
| SMTP_SERVER | ✅ | str | 发送端邮箱的 SMTP 服务器。 | smtp.qq.com |
| SMTP_PORT | ✅ | int | SMTP 端口（SSL 推荐 465）。 | 465 |
| SENDER | ✅ | str | 发送端邮箱账号。 | researcher@qq.com |
| SENDER_PASSWORD | ✅ | str | 邮箱授权码（非登录密码）。 | abcdefghijklmnop |
| RECEIVER | ✅ | str | 接收论文列表的邮箱。 | target@outlook.com |
| USE_LLM_API | | bool | 是否使用云端 API (默认为 False)。 | False |
| OPENAI_API_KEY | | str | API Key (若使用 API 则必填)。 | sk-xxxxxxxx |

3. **设置 Repository Variables** (公开变量，方便随时修改)：

| 变量名 | 必填 | 类型 | 描述 | 示例 |
| :--- | :--- | :--- | :--- | :--- |
| RETRIEVER_TARGET | ✅ | str | **核心兴趣定义**，每行一个。 | 见下文示例 |
| MAX_PAPER_NUM | | int | 邮件展示的最大论文数量（默认 100）。 | 20 |
| LANGUAGE | | str | TLDR 生成的语言(默认为English) | Chinese |
| SEND_EMPTY | | bool | 若当日无匹配论文是否发送空邮件。 | False |

### 🧠 兴趣定义 (RETRIEVER_TARGET) 最佳实践

在 `RETRIEVER_TARGET` 变量中详细定义您的研究兴趣（短句或关键词），这直接影响 LLM 的判别效果：

```text
Reinforcement learning and preference alignment (RLHF) for LLMs
Architecture and parameter efficiency optimization for Transformers
Autonomous agents, planning, and multi-step reasoning systems
Advanced Retrieval-Augmented Generation (RAG) and knowledge integration
LLM
Edge Detection
```
> 如果大模型使用关键词来进行相似度检索效果会约等于直接用关键字检索，建议使用**中等颗粒度**的**英文描述**（检索模型为英文语料模型）

## 💻 本地运行

本项目支持 [uv](https://github.com/astral-sh/uv) 驱动，可以轻松在本地环境运行：

```bash
# 设置必要的环境变量
# export ARXIV_QUERY=cs.AI+cs.CV
# ... 

# 运行项目
uv run main.py
```

> [!IMPORTANT]
> 运行本地模型将自动下载 **Qwen2.5-3B-Instruct/Llama3.2-3B-Instruct** (约 3GB)。请确保您的网络环境稳定且内存不少于 8GB。

## 📖 工作原理

1. **获取数据**: 每日定时通过 ArXiv API 抓取指定分类下的最新论文摘要。
2. **属性判别**: 本地 LLM 利用 **少样本 + 思维链（CoT）** 逻辑分析摘要，识别论文细节。
3. **语义评分**: 计算论文贡献与 `RETRIEVER_TARGET` 的语义契合度，给出 0-100 的相关度评分。
4. **生成总结**: 提取论文核心点，生成精简的 TLDR。
5. **富文本推送**: 渲染 HTML 模板并通过 SMTP 发送。

## 🛠️ 技术栈

- **Python 3.10+**
- **llama-cpp-python**: 本地推理后端支持
- **sentence-transformers**: 语义向量计算
- **arxiv**: ArXiv API 客户端
- **GitHub Actions**: 自动化工作流调度

## 👯‍♂️ 参与贡献

欢迎提交Issue和Pull Request！我们鼓励：
- 改进推荐算法
- 添加新功能
- 修复bug
- 优化文档

所有功能性更新请务必先合并至 `dev` 分支。

## 📝 许可证

本项目基于 [MIT License](https://github.com/yourusername/arxiv_radar/blob/main/LICENSE) 协议。

## 🙏 致谢

- [arxiv.org](https://arxiv.org/) 提供开放的科研数据。
- [Llama.cpp](https://github.com/ggerganov/llama.cpp) 提供本地模型运行支持。
- [Zotero-arXiv-Daily](https://github.com/TideDra/zotero-arxiv-daily) 提供的灵感。

**开始使用ArXiv Radar，让最新的研究成果自动找到您！** 🚀
