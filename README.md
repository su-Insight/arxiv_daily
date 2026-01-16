# ArXiv Radar

[中文版本 (Chinese Version)](README_zh.md)

ArXiv Radar是一个自动化工具，用于每日从ArXiv检索最新论文，使用本地LLM解析摘要，计算与用户定义关键字的相似度，并将最相关的前K篇论文通过邮件发送给用户。

## 功能特点

- 📚 **每日更新**：自动从ArXiv获取过去24小时内的最新论文
- 🔍 **智能检索**：支持多关键字/短语检索，可选择ArXiv分类
- 🧠 **本地LLM**：使用本地部署的LLM解析和理解论文摘要，保护隐私
- ⚡ **相似度计算**：基于余弦相似度算法筛选最相关的论文
- 📧 **邮件推送**：将筛选后的论文列表发送到指定邮箱
- ⏰ **定时执行**：可配置的每日执行时间
- 📁 **轻量级设计**：仅包含4个核心文件，易于部署和使用

## 项目结构

```
arxiv_radar/
├── main.py                  # Main program file containing all core logic
├── src/
│   ├── llm.py              # LLM-related functions and utilities
│   ├── paper.py            # Paper data structure and ArXiv API interactions
│   ├── rerank.py           # Paper reranking logic using LLM
│   └── construct_email.py  # Email construction and sending functions
├── requirements.txt         # Dependencies list
├── README.md               # Project documentation (English)
├── README_zh.md            # Project documentation (Chinese)
└── .github/
    └── workflows/          # GitHub Actions workflows
        └── main.yml        # Daily execution workflow
```

## 安装步骤

### 1. 环境要求

- Python 3.7+
- Sufficient disk space (for storing LLM models, approximately 100MB-2GB depending on model size)

### 2. 安装依赖

```bash
# Install project dependencies
pip install -r requirements.txt
```

## 配置

### 1. GitHub Actions Secrets and Variables

Configure the following secrets and variables in your GitHub repository:

**Secrets:**
- `ARXIV_QUERY`: ArXiv search query
- `SMTP_SERVER`: SMTP server address
- `SMTP_PORT`: SMTP server port
- `SENDER`: Sender email address
- `RECEIVER`: Receiver email address
- `SENDER_PASSWORD`: Sender email password
- `USE_LLM_API`: Whether to use OpenAI API (true/false)
- `OPENAI_API_KEY`: OpenAI API key (required if USE_LLM_API is true)
- `OPENAI_API_BASE`: OpenAI API base URL (optional)
- `MODEL_NAME`: OpenAI model name (optional, default: gpt-4o)

**Variables:**
- `REPOSITORY`: Repository name (default: your GitHub username/arxiv_radar)
- `REF`: Branch name (default: main)
- `SEND_EMPTY`: Whether to send empty email when no papers found (true/false)
- `MAX_PAPER_NUM`: Maximum number of papers to recommend
- `RETRIEVER_TARGET`: Interest domains, one per line
- `LANGUAGE`: Language for TLDR generation (default: English)

## Usage

### Run Locally

```bash
python main.py
```

### GitHub Actions (Recommended)

1. Fork this repository
2. Configure secrets and variables as described above
3. The workflow will run automatically daily at the scheduled time

## Troubleshooting

### 1. Failed to connect to SMTP server

**Solutions:**
- Check SMTP server address and port correctness
- Ensure TLS/SSL encryption is properly configured
- Verify email username and password
- For Gmail users, enable "Less secure app access" or use app-specific passwords

### 2. Model loading failure

**Solutions:**
- Check network connection to ensure model can be downloaded
- Try using a smaller model
- Manually download the model and specify local path

### 3. No relevant papers found

**Solutions:**
- Check keyword correctness, try using broader keywords
- Reduce ArXiv category restrictions
- Increase `MAX_PAPER_NUM` parameter value

### 4. Slow execution

**Solutions:**
- Use a smaller LLM model
- Reduce `MAX_PAPER_NUM` parameter value
- Consider running in an environment with GPU (sentence-transformers supports GPU acceleration)

## Technology Stack

- **Python 3.7+**: Main development language
- **arxiv**: ArXiv API client
- **llama_cpp**: Local LLM integration
- **openai**: OpenAI API integration (optional)
- **sentence-transformers**: Text embedding generation
- **scikit-learn**: Similarity calculation
- **schedule**: Scheduled task management
- **smtplib**: Email sending
- **GitHub Actions**: Continuous integration and deployment

## License

MIT License

## Contributing

Issue and Pull Request are welcome!

## Changelog

### v1.0.0 (2024-01-15)
- Initial version release
- Implemented core features: ArXiv retrieval, local LLM parsing, similarity calculation, email sending, and scheduled execution

---

**Note**: When running for the first time, the program will automatically download the specified LLM model to the local `./models` directory. Ensure network connection is available.