import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from openai import OpenAI
import os

# ── 配置 ──────────────────────────────────────────────
CATEGORIES = ['cs.RO', 'cs.AI']
MIN_PAPERS = 10
MAX_PAPERS = 20
NEWAPI_BASE_URL = "https://hone.vvvv.ee/"
NEWAPI_MODEL = "claude-sonnet-4-6"
# ──────────────────────────────────────────────────────

def fetch_arxiv_papers(categories, max_results=200):
    query = "+OR+".join([f"cat:{cat}" for cat in categories])
    url = (f"http://export.arxiv.org/api/query"
           f"?search_query={query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(response.text)
    beijing_tz = timezone(timedelta(hours=8))
    cutoff = datetime.now(beijing_tz).date() - timedelta(days=2)

    papers = []
    for entry in root.findall('atom:entry', ns):
        published = entry.find('atom:published', ns).text
        pub_date = datetime.fromisoformat(published.replace('Z', '+00:00')).astimezone(beijing_tz).date()
        if pub_date < cutoff:
            break
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        link = entry.find('atom:id', ns).text.strip()
        cats = [c.get('term') for c in entry.findall('atom:category', ns)]
        papers.append({'title': title, 'abstract': abstract, 'link': link, 'categories': cats})

    return papers


def stage1_filter(papers, deepseek_key):
    """阶段1：DeepSeek 筛选相关论文，返回论文子集（10-20篇）"""
    client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com/v1")

    papers_text = ""
    for i, p in enumerate(papers, 1):
        papers_text += f"{i}. {p['title']}\n摘要：{p['abstract'][:200]}\n\n"

    prompt = f"""你是无人机导航与世界模型方向的科研助手。

重点关注方向：
- 世界模型（World Model）用于规划/导航/控制
- 无人机/UAV自主导航、运动规划
- 视觉语言导航（VLN）
- 强化学习 + 机器人/无人机
世界模型方面领域的论文权重最高，VLN的权重最低

以下是今日{len(papers)}篇论文。请按相关度筛选出{MIN_PAPERS}-{MAX_PAPERS}篇最相关的论文。
要求：
- 强相关的必须保留
- 弱相关但有参考价值的酌情保留，保证总数不少于{MIN_PAPERS}篇
- 只输出编号列表，每行一个数字，不要其他内容

论文列表：
{papers_text}

只输出编号，例如：
3
7
12
19"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )

    # 解析返回的编号
    raw = response.choices[0].message.content.strip()
    indices = []
    for line in raw.splitlines():
        line = line.strip().strip('.-')
        if line.isdigit():
            idx = int(line) - 1
            if 0 <= idx < len(papers):
                indices.append(idx)

    selected = [papers[i] for i in indices]

    # 保底：如果返回数量不足，补充到MIN_PAPERS篇
    if len(selected) < MIN_PAPERS:
        print(f"筛选结果{len(selected)}篇不足{MIN_PAPERS}篇，补充至{MIN_PAPERS}篇")
        existing_indices = set(indices)
        for i in range(len(papers)):
            if i not in existing_indices:
                selected.append(papers[i])
            if len(selected) >= MIN_PAPERS:
                break

    print(f"阶段1筛选完成：{len(papers)}篇 → {len(selected)}篇")
    return selected[:MAX_PAPERS]


def stage2_summarize(papers, deepseek_key):
    """阶段2：逐篇分析总结，带fallback"""
    newapi_keys = [
        os.environ.get('NEWAPI_KEY_1'),
        os.environ.get('NEWAPI_KEY_2'),
        os.environ.get('NEWAPI_KEY_3'),
    ]

    papers_text = ""
    for i, p in enumerate(papers, 1):
        cats = ', '.join(p['categories'][:2])
        papers_text += f"{i}. [{cats}] {p['title']}\n摘要：{p['abstract'][:300]}\n链接：{p['link']}\n\n"

    prompt = f"""对以下{len(papers)}篇论文，每篇给出3-4句话的中文小总结，格式如下：

**1. 论文标题**
🔥/👀 （强相关用🔥，有参考价值用👀）
总结：xxx（方法是什么、解决什么问题、核心贡献）
链接：xxx

论文列表：
{papers_text}"""

    # 依次尝试3个 New API key
    last_error = None
    for i, key in enumerate(newapi_keys, 1):
        if not key:
            continue
        try:
            print(f"阶段2：尝试 New API key {i}...")
            client = OpenAI(api_key=key, base_url=NEWAPI_BASE_URL)
            response = client.chat.completions.create(
                model=NEWAPI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                timeout=90,
            )
            print(f"New API key {i} 成功")
            return response.choices[0].message.content
        except Exception as e:
            print(f"New API key {i} 失败：{e}")
            last_error = e

    # 所有 New API 失败，回退到 DeepSeek
    print(f"所有 New API 均失败（{last_error}），回退到 DeepSeek...")
    client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com/v1")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
        timeout=90,
    )
    print("DeepSeek 回退成功")
    return response.choices[0].message.content


def send_to_wechat(title, content, sct_key):
    url = f"https://sctapi.ftqq.com/{sct_key}.send"
    if len(content) > 4000:
        content = content[:4000] + "\n\n...（已截断）"
    resp = requests.post(url, data={'title': title, 'desp': content}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def main():
    sct_key = os.environ['SCT_KEY']
    deepseek_key = os.environ['DEEPSEEK_API_KEY']
    beijing_tz = timezone(timedelta(hours=8))
    today_str = datetime.now(beijing_tz).strftime('%Y-%m-%d')

    print("拉取arxiv论文中...")
    papers = fetch_arxiv_papers(CATEGORIES)
    print(f"共获取 {len(papers)} 篇")

    if not papers:
        send_to_wechat(f"arxiv日报 {today_str}", "今日暂无新论文", sct_key)
        return

    selected = stage1_filter(papers, deepseek_key)
    summary = stage2_summarize(selected, deepseek_key)

    title = f"arxiv日报 {today_str} | 精选{len(selected)}篇"
    print("推送微信中...")
    result = send_to_wechat(title, summary, sct_key)
    print(f"完成：{result}")


if __name__ == "__main__":
    main()
