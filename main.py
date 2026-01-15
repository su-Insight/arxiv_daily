if __name__ == '__main__':
    add_argument('--send_empty', type=bool, help='If get no arxiv paper, send empty email',default=False)
    add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend',default=100)
    add_argument('--arxiv_query', type=str, help='Arxiv search query')
    add_argument('--smtp_server', type=str, help='SMTP server')
    add_argument('--smtp_port', type=int, help='SMTP port')
    add_argument('--sender', type=str, help='Sender email address')
    add_argument('--receiver', type=str, help='Receiver email address')
    add_argument('--sender_password', type=str, help='Sender email password')
    add_argument(
        "--use_llm_api",
        type=bool,
        help="Use OpenAI API to generate TLDR",
        default=False,
    )
    add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        default=None,
    )
    add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://api.openai.com/v1",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="gpt-4o",
    )
    add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
        default="English",
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    print(args)
    # assert (
    #     not args.use_llm_api or args.openai_api_key is not None
    # )  # If use_llm_api is True, openai_api_key must be provided
    # if args.debug:
    #     logger.remove()
    #     logger.add(sys.stdout, level="DEBUG")
    #     logger.debug("Debug mode is on.")
    # else:
    #     logger.remove()
    #     logger.add(sys.stdout, level="INFO")

    # logger.info("Retrieving Zotero corpus...")
    # corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    # logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    # if args.zotero_ignore:
    #     logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
    #     corpus = filter_corpus(corpus, args.zotero_ignore)
    #     logger.info(f"Remaining {len(corpus)} papers after filtering.")
    # logger.info("Retrieving Arxiv papers...")
    # papers = get_arxiv_paper(args.arxiv_query, args.debug)
    # if len(papers) == 0:
    #     logger.info("No new papers found. Yesterday maybe a holiday and no one submit their work :). If this is not the case, please check the ARXIV_QUERY.")
    #     if not args.send_empty:
    #       exit(0)
    # else:
    #     logger.info("Reranking papers...")
    #     papers = rerank_paper(papers, corpus)
    #     if args.max_paper_num != -1:
    #         papers = papers[:args.max_paper_num]
    #     if args.use_llm_api:
    #         logger.info("Using OpenAI API as global LLM.")
    #         set_global_llm(api_key=args.openai_api_key, base_url=args.openai_api_base, model=args.model_name, lang=args.language)
    #     else:
    #         logger.info("Using Local LLM as global LLM.")
    #         set_global_llm(lang=args.language)

    # html = render_email(papers)
    # logger.info("Sending email...")
    # send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)
    # logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")