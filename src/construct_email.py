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
    .filter-buttons {
      text-align: right;
      margin-bottom: 16px;
      padding: 8px;
      background-color: #f0f0f0;
      border-radius: 8px;
    }
    .filter-btn {
      display: inline-block;
      text-decoration: none;
      font-size: 14px; font-weight: bold;
      color: #fff; background-color: #5bc0de;
      padding: 8px 16px; border-radius: 4px;
      margin-left: 8px;
      cursor: pointer;
      border: none;
      outline: none;
    }
    .filter-btn.active {
      background-color: #337ab7;
    }
    .paper-block {
      margin-bottom: 16px;
    }
  </style>
</head>
<body>

<!-- Domain filter buttons will be inserted here -->
<div class="filter-buttons">
  <button class="filter-btn active" onclick="filterByDomain('all')">All Domains</button>
  <!-- Additional domain buttons will be generated dynamically -->
</div>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

<script>
function filterByDomain(domain) {
  // Get all paper blocks
  const paperBlocks = document.querySelectorAll('.paper-block');
  
  // Update button states
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  document.querySelector(`[onclick="filterByDomain('${domain}')"]`).classList.add('active');
  
  // Filter papers
  paperBlocks.forEach(block => {
    if (domain === 'all' || block.classList.contains(`domain-${domain}`)) {
      block.style.display = 'block';
    } else {
      block.style.display = 'none';
    }
  });
}
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

def get_block_html(title:str, authors:str, rate:str, score:float,arxiv_id:str, abstract:str, pdf_url:str, code_url:str=None, affiliations:str=None):
    code = f'<a href="{code_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #5bc0de; padding: 8px 16px; border-radius: 4px; margin-left: 8px;">Code</a>' if code_url else ''
    block_template = """
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
            <strong>Relevance:</strong> {rate} <span style="font-size: 12px; color: #888;">({score:.1f}/100)</span>
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
    return block_template.format(title=title, authors=authors,rate=rate, score=score, arxiv_id=arxiv_id, abstract=abstract, pdf_url=pdf_url, code=code, affiliations=affiliations)

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
    # Get unique domains from interests or default to empty list
    domains = []
    if interests:
        domains = [interest.strip() for interest in interests if interest.strip()]
    
    # Generate domain filter buttons
    domain_buttons = ''
    for domain in domains:
        # Replace spaces with underscores for CSS class
        domain_class = domain.replace(' ', '_').replace('-', '_')
        domain_buttons += f'<button class="filter-btn" onclick="filterByDomain(\'{domain_class}\')">{domain}</button>'
    
    parts = []
    if len(papers) == 0 :
        # Add domain buttons to empty state
        framework_with_buttons = framework.replace('<!-- Additional domain buttons will be generated dynamically -->', domain_buttons)
        return framework_with_buttons.replace('__CONTENT__', get_empty_html())
    
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
        
        # Get the domain classes for this paper
        domain_classes = ''
        if domains and hasattr(p, 'score') and p.score > 0:
            # For now, we'll just add all domains - in a real implementation, you'd want to track which domain(s) the paper matched
            for domain in domains:
                domain_class = domain.replace(' ', '_').replace('-', '_')
                domain_classes += f' domain-{domain_class}'
        
        parts.append(get_block_html(p.title, authors, rate, p.score, p.arxiv_id, p.tldr, p.pdf_url, p.code_url, affiliations))
        time.sleep(10)

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    
    # Add domain buttons to framework
    framework_with_buttons = framework.replace('<!-- Additional domain buttons will be generated dynamically -->', domain_buttons)
    return framework_with_buttons.replace('__CONTENT__', content)

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