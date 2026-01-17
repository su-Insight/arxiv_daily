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

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

<p align="center"> 🚀 Daily intelligent retrieval, semantic reranking, and push notifications for ArXiv papers based on your research interests using local LLMs.
    <br> 
</p>

> [!IMPORTANT]
> This project is under active development. Please **Watch** this repository and **Sync** your Forked repository timely when the upstream updates to enjoy the latest semantic discrimination logic and bug fixes.

## 🧐 About <a name = "about"></a>

> Want to stay at the forefront of research? Simply **Fork** (and **Star**⭐) this repo to launch your exclusive Paper Radar! 😊

**ArXiv Radar** is an automated tool designed to precisely capture your research interests. Unlike simple keyword searches, it utilizes local lightweight Large Language Models (Llama3.2-3B) for **Deep Semantic Discrimination** of daily papers. The system analyzes the core contributions, distinguishing "hardcore technical innovations" from "routine applications," ensuring the content pushed to your email is truly valuable research.

Powered by **GitHub Actions**, it provides a **zero-cost**, **zero-installation**, and **fully automated** daily academic newsletter service.

## ✨ Features

- **Completely Free**: All computing tasks are completed within the free quota provided by GitHub Actions; no private server required.
- **Semantic Reranking**: Leverages LLM Few-shot and Chain-of-Thought (CoT) capabilities to score papers, effectively filtering information noise.
- **AI-Generated TLDR**: Generates concise summaries for each paper (supporting Chinese), helping you screen core information in seconds.
- **Rich Text Email**: Integrated direct PDF links, code repository addresses, author affiliations, and high-score theme styles, with support for interactive filtering in emails.
- **Local Inference**: Downloads and runs models locally by default, protecting your research interest privacy without the need for API fees.
- **Optional API Support**: If you seek faster generation speeds or higher quality, cloud API access is also supported.

## 📷 Preview

<div align="left">
  
| <img width="882" height="697" alt="image" src="https://github.com/user-attachments/assets/a4ded5b7-b673-4e7c-a882-1c2ef2f47a21" /> |
| :---: |

</div>

## 🚀 Usage Guide

### Quick Start

1. **Fork this repository** (Don't forget to Star 😉)

<div align="left">
  
| <img width="839" height="634" alt="image" src="https://github.com/user-attachments/assets/95d6c54a-e781-4cd5-8a33-9f2f6d5b1ace" /> |
| :---: |

</div>

2. **Set GitHub Secrets**:
   Go to your repository `Settings -> Secrets and variables -> Actions`, and add the following `Repository secrets`:
   
<div align="left">
  
| <img width="845" height="572" alt="image" src="https://github.com/user-attachments/assets/b3a4b5f6-a2af-4786-b1dd-5d6b7cb170bc"  /> |
| :---: |

</div>

| Variable Name | Required | Type | Description | Example |
| :--- | :---: | :--- | :--- | :--- |
| ARXIV_QUERY | ✅ | str | ArXiv categories to retrieve, connected by `+`. [Details here](https://arxiv.org/category_taxonomy). | cs.AI+cs.CV+cs.LG |
| SMTP_SERVER | ✅ | str | SMTP server of the sender email. | smtp.qq.com |
| SMTP_PORT | ✅ | int | SMTP port (465 is recommended for SSL). | 465 |
| SENDER | ✅ | str | Sender email account. | researcher@qq.com |
| SENDER_PASSWORD | ✅ | str | Email authorization code (not login password). | abcdefghijklmnop |
| RECEIVER | ✅ | str | Email to receive the paper list. | target@outlook.com |
| USE_LLM_API | | bool | Whether to use cloud API (Default is False). | False |
| OPENAI_API_KEY | | str | API Key (Required if using API). | sk-xxxxxxxx |
| OPENAI_API_BASE | | str | API URL when using the API to access LLMs. If not filled in, the default is the OpenAI URL. | https://api.siliconflow.cn/v1 |
| MODEL_NAME | | str | Model name when using the API to access LLMs. If not filled in, the default is gpt-4o. Qwen/Qwen2.5-7B-Instruct is recommended when using SiliconFlow. | Qwen/Qwen2.5-7B-Instruct |
> For SMTP application, search directly for the corresponding email + SMTP on the web.

3. **Set Repository Variables** (Public variables, convenient for modifications):

<div align="left">
  
| <img width="853" height="608" alt="image" src="https://github.com/user-attachments/assets/7476b989-8c7b-4fc4-b494-2785e6de8117" /> |
| :---: |

</div>

| Variable Name | Required | Type | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| RETRIEVER_TARGET | ✅ | str | **Core Interest Definitions**, one per line. | See examples below |
| MAX_PAPER_NUM | | int | Max number of papers displayed in email (Default 100). | 20 |
| LANGUAGE | | str | Language for TLDR generation (Default is English). | Chinese |
| SEND_EMPTY | | bool | Whether to send an empty email if no papers match. | False |

### 🧠 Interest Definition (RETRIEVER_TARGET) Best Practice

Define your research interests in detail (phrases or keywords) in the `RETRIEVER_TARGET` variable, which directly affects the LLM discrimination:

```text
Reinforcement learning and preference alignment (RLHF) for LLMs
Architecture and parameter efficiency optimization for Transformers
Autonomous agents, planning, and multi-step reasoning systems
Advanced Retrieval-Augmented Generation (RAG) and knowledge integration
LLM
Edge Detection
```

> If the LLM uses keywords for similarity retrieval, the effect is roughly equal to a direct keyword search. It is recommended to use **medium-grained English descriptions** (the retrieval model is an English corpus model).

### 🔄 Auto-Sync and Update

> [!TIP]
> **No manual synchronization required!** By configuring the following variables, the workflow will automatically pull the latest code from the specified repository before each run, ensuring you always enjoy the latest algorithm optimizations.

If you want your Forked repository to stay consistent with the upstream, please set the following in **Repository Variables**:

| Key | Required | Description | Example |
| :--- | :---: | :--- | :--- |
| **REPOSITORY** | ✅ | Repository providing workflow source code. It will pull the latest code automatically. | `su-Insight/arxiv-radar` |
| **REF** | ✅ | Specifies the code branch to run (`main` for stable, `dev` for preview) | `main` |

---

## 🧪 Testing and Verification (Testing) <a name = "testing"></a>

Before officially starting the daily automatic push, you can verify the configuration by manually triggering the test workflow:

1. Go to the **Actions** tab of your Forked repository.
2. Select **`Test-Daily Paper Sender`** in the left list.
3. Click the **`Run workflow`** dropdown on the right, and click the green **`Run workflow`** button.

<div align="left">

| <img width="893" height="247" alt="image" src="https://github.com/user-attachments/assets/7aba0dc4-d1d8-413c-9d22-c77f2d8da110" /> |
| :---: |

</div>



> [!NOTE]
> **Test Version vs Formal Version**:
> - **Test-Workflow**: Specifically for debugging. It ignores date limits and retrieves a fixed set of 5 ArXiv papers, allowing you to check email layout and LLM scoring results immediately.
> - **Main-Workflow**: Triggers automatically at 20:00 UTC daily, retrieving only new papers published in the last 24 hours.

## 💻 Local Execution

This project supports [uv](https://github.com/astral-sh/uv) and can be easily run in a local environment:

```bash
# Set necessary environment variables
# export ARXIV_QUERY=cs.AI+cs.CV
# ... 

# Run the project
uv run main.py
```

> [!IMPORTANT]
> Running the local model will automatically download **Qwen2.5-3B-Instruct/Llama3.2-3B-Instruct** (approx. 3GB). Please ensure your network environment is stable and you have at least 8GB of RAM.

## 📖 How It Works

1. **Get Data**: Daily scheduled crawl of the latest paper abstracts under specified categories via ArXiv API.
2. **Attribute Discrimination**: Local LLM uses **Few-shot + Chain-of-Thought (CoT)** logic to analyze abstracts and identify paper details.
3. **Semantic Scoring**: Calculates the semantic fit between the paper's contribution and `RETRIEVER_TARGET`, providing a relevance score of 0-100.
4. **Generate Summary**: Extracts the core points of the paper and generates a concise TLDR.
5. **Rich Text Push**: Renders the HTML template and sends it via SMTP.

## 🛠️ Tech Stack

- **Python 3.10+**
- **llama-cpp-python**: Local inference backend support
- **sentence-transformers**: Semantic vector computation
- **arxiv**: ArXiv API client
- **GitHub Actions**: Automated workflow scheduling

## 👯‍♂️ Contributing

Issues and Pull Requests are welcome! We encourage:

- Improving recommendation algorithms
- Adding new features
- Fixing bugs
- Optimizing documentation

All functional updates must be merged into the `dev` branch first.

## 📝 License

This project is based on the [Apache-2.0 license](https://github.com/su-Insight/arxiv_radar/blob/main/LICENSE) protocol.

## 🙏 Acknowledgements

- [arxiv.org](https://arxiv.org/) for providing open research data.
- [Llama.cpp](https://github.com/ggerganov/llama.cpp) for local model execution support.
- [Zotero-arXiv-Daily](https://github.com/TideDra/zotero-arxiv-daily) for inspiration.

**Start using ArXiv Radar and let the latest research find you automatically!** 🚀
