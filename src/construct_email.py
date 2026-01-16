from .paper import ArxivPaper
import math
from tqdm import tqdm
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import smtplib
import datetime
import time
from loguru import logger

framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
    .paper-block {
      margin-bottom: 16px;
    }
    .interest-tag {
      cursor: pointer;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

<script>
// 自适应调整兴趣标签宽度的函数
function adjustInterestTags() {
  // 遍历所有论文区块
  document.querySelectorAll('.paper-block').forEach(block => {
    const relevanceRow = block.querySelector('tr:nth-child(3) td');
    if (!relevanceRow) return;
    
    const interestTags = relevanceRow.querySelectorAll('.interest-tag');
    if (interestTags.length === 0) return;
    
    // 获取相关性行的可用宽度（减去其他内容的宽度）
    const relevanceText = relevanceRow.querySelector('strong');
    const stars = relevanceRow.querySelector('.star-wrapper');
    const scoreText = relevanceRow.querySelector('span:nth-child(3)');
    
    let otherContentWidth = 0;
    if (relevanceText) otherContentWidth += relevanceText.offsetWidth;
    if (stars) otherContentWidth += stars.offsetWidth;
    if (scoreText) otherContentWidth += scoreText.offsetWidth;
    
    // 添加一些边距和间距
    otherContentWidth += 40; // 边距和间距
    
    const rowWidth = relevanceRow.offsetWidth;
    const availableWidth = rowWidth - otherContentWidth;
    
    if (availableWidth <= 0) return;
    
    // 计算每个标签的最大宽度
    const minTagWidth = 50; // 每个标签的最小宽度
    const maxTagWidth = 150; // 每个标签的最大宽度
    
    if (availableWidth / interestTags.length >= maxTagWidth) {
      // 如果有足够的空间，每个标签都可以显示最大宽度
      interestTags.forEach(tag => {
        tag.style.maxWidth = maxTagWidth + 'px';
      });
    } else if (availableWidth / interestTags.length <= minTagWidth) {
      // 如果空间不足，每个标签都显示最小宽度
      interestTags.forEach(tag => {
        tag.style.maxWidth = minTagWidth + 'px';
      });
    } else {
      // 否则，平均分配可用空间
      const avgWidth = Math.floor(availableWidth / interestTags.length);
      interestTags.forEach(tag => {
        tag.style.maxWidth = avgWidth + 'px';
      });
    }
  });
}

// 页面加载完成后执行
window.addEventListener('load', adjustInterestTags);

// 窗口大小改变时重新调整
window.addEventListener('resize', adjustInterestTags);
</script>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
  return block_template

def get_block_html(title:str, authors:str, rate:str, score:float,arxiv_id:str, abstract:str, pdf_url:str, code_url:str=None, affiliations:str=None, high_score_interests:list=None):
    code = f'<a href="{code_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #5bc0de; padding: 8px 16px; border-radius: 4px; margin-left: 8px;">Code</a>' if code_url else ''
    
    # 导入截断函数
    from src.rerank import truncate_interest
    
    # 生成多个interest标签
    interest_tags = ''
    if high_score_interests and isinstance(high_score_interests, list):
        for interest in high_score_interests:
            # 在显示时截断文本，但title属性使用完整文本
            truncated_interest = truncate_interest(interest)
            interest_tags += f'<span class="interest-tag" data-full-interest="{interest}" title="{interest}" style="display: inline-block; background-color: #4CAF50; color: white; padding: 3px 6px; border-radius: 10px; font-size: 10px; font-weight: bold; margin-left: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 120px; line-height: 1; box-sizing: border-box;">{truncated_interest}</span>'
    
    block_template = f"""
    <div class="paper-block">
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate} <span style="font-size: 12px; color: #888;">({score:.0f}/100)</span>{interest_tags}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>arXiv ID:</strong> <a href="https://arxiv.org/abs/{arxiv_id}" target="_blank">{arxiv_id}</a>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {abstract}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
            {code}
        </td>
    </tr>
</table>
    </div>
"""
    # return block_template.format(title=title, authors=authors,rate=rate, score=score, arxiv_id=arxiv_id, abstract=abstract, pdf_url=pdf_url, code=code, affiliations=affiliations, high_score_interests=high_score_interests)
    return block_template

def get_stars(score:float):
    # Convert percentage score to original scale (divide by 10)
    adjusted_score = score / 10
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if adjusted_score <= low:
        return ''
    elif adjusted_score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((adjusted_score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'


def render_email(papers:list[ArxivPaper], interests:list[str]=None):
    parts = []
    if len(papers) == 0 :
        return framework.replace('__CONTENT__', get_empty_html())
    
    for p in tqdm(papers,desc='Rendering Email'):
        rate = get_stars(p.score)
        author_list = [a.name for a in p.authors]
        num_authors = len(author_list)
        
        if num_authors <= 5:
            authors = ', '.join(author_list)
        else:
            authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])
        if p.affiliations is not None:
            affiliations = p.affiliations[:5]
            affiliations = ', '.join(affiliations)
            if len(p.affiliations) > 5:
                affiliations += ', ...'
        else:
            affiliations = 'Unknown Affiliation'
        
        parts.append(get_block_html(p.title, authors, rate, p.score, p.arxiv_id, p.tldr, p.pdf_url, p.code_url, affiliations, p.high_score_interests))
        time.sleep(10)

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    
    return framework.replace('__CONTENT__', content)

def send_email(sender:str, receiver:str, password:str,smtp_server:str,smtp_port:int, html:str,):
    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    msg = MIMEText(html, 'html', 'utf-8')
    msg['From'] = _format_addr('Github Action <%s>' % sender)
    msg['To'] = _format_addr('You <%s>' % receiver)
    today = datetime.datetime.now().strftime('%Y/%m/%d')
    msg['Subject'] = Header(f'Daily arXiv {today}', 'utf-8').encode()

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
    except Exception as e:
        logger.warning(f"Failed to use TLS. {e}")
        logger.warning(f"Try to use SSL.")
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)

    server.login(sender, password)
    server.sendmail(sender, [receiver], msg.as_string())
    server.quit()