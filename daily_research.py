"""
自动驾驶 VLA 模型 / 世界模型 / 具身智能 每日研究简报
数据源: arXiv + GitHub Trending
摘要生成: Google Gemini API
"""

import os
import sys
import json
import ssl
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import re
import time

# ============================================================
# 配置
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
FEISHU_WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK_URL", "")

# arXiv 搜索关键词配置: 使用 ti:(标题) + abs:(摘要) 精确搜索
ARXIV_TOPICS = {
    "自动驾驶 VLA / 端到端": [
        'ti:"vision-language-action" OR abs:"vision-language-action"',
        'ti:"VLA" AND (ti:"driving" OR abs:"driving" OR ti:"autonomous" OR abs:"autonomous")',
        'ti:"end-to-end" AND ti:"driving" AND (ti:"language" OR abs:"language model" OR ti:"LLM")',
        'ti:"driving" AND abs:"large language model" AND abs:"autonomous"',
    ],
    "世界模型 / 生成式仿真": [
        'ti:"world model" AND (ti:"driving" OR abs:"driving" OR ti:"autonomous")',
        'ti:"world model" AND (abs:"video prediction" OR abs:"generation")',
        'ti:"world model" AND (abs:"reinforcement learning" OR abs:"decision making")',
        'ti:"generative" AND ti:"world model"',
    ],
    "自动驾驶仿真 / 数字孪生": [
        'ti:"simulation" AND (ti:"driving" OR abs:"autonomous driving" OR ti:"autonomous")',
        'ti:"digital twin" AND (ti:"driving" OR ti:"autonomous")',
        'ti:"neural simulation" AND (abs:"driving" OR abs:"autonomous")',
        'ti:"simulator" AND (abs:"autonomous" OR abs:"driving") AND abs:"learning"',
    ],
    "具身智能": [
        'ti:"embodied" AND (ti:"robot" OR abs:"robot" OR ti:"manipulation" OR abs:"manipulation")',
        'ti:"embodied" AND (ti:"agent" OR abs:"navigation" OR ti:"instruction")',
        'ti:"humanoid" AND (abs:"learning" OR abs:"control" OR abs:"locomotion")',
        'ti:"robot" AND abs:"foundation model" AND (abs:"manipulation" OR abs:"navigation")',
    ],
}

# GitHub 搜索关键词
GITHUB_TOPICS = {
    "自动驾驶": ["autonomous driving", "self-driving car"],
    "世界模型 / 仿真": ["world model simulation", "generative world model"],
    "仿真引擎": ["driving simulator", "autonomous driving simulation"],
    "具身智能": ["embodied AI robot", "embodied intelligence"],
}

# Google News RSS 搜索关键词
NEWS_TOPICS = {
    "自动驾驶 / 机器人出租车": [
        "autonomous driving self-driving car robotaxi",
        "Waymo Cruise driverless",
        "Tesla FSD autonomous robotaxi",
    ],
    "世界模型 / 生成式 AI": [
        "world model AI robot",
        "generative world model simulation",
        "NVIDIA world model foundation",
    ],
    "具身智能 / 人形机器人": [
        "humanoid robot Figure 1 Optimus",
        "embodied AI robot Boston Dynamics",
        "Unitree humanoid robot",
    ],
    "自动驾驶仿真 / 数字孪生": [
        "autonomous driving simulation digital twin",
        "driving simulator NVIDIA DRIVE Sim",
        "CARLA simulator autonomous",
    ],
}

ARXIV_MAX_RESULTS = 8  # 每个关键词最多返回论文数
REPORT_DIR = os.environ.get("REPORT_DIR", "reports")

# SSL 上下文：本地开发时跳过证书验证，GitHub Actions 环境不受影响
_SSL_CTX = ssl.create_default_context()
if os.environ.get("PYTHONHTTPSVERIFY", "1") != "1":
    _SSL_CTX.check_hostname = False
    _SSL_CTX.verify_mode = ssl.CERT_NONE

# ============================================================
# arXiv 爬取
# ============================================================

def fetch_arxiv_papers(query, max_results=ARXIV_MAX_RESULTS):
    """从 arXiv API 搜索论文"""
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    }
    url = base_url + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AutoResearchBot/1.0"})
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
            data = resp.read().decode("utf-8")
    except Exception as e:
        print(f"  [WARN] arXiv 请求失败 ({query}): {e}", file=sys.stderr)
        return []

    papers = []
    root = ET.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        title = re.sub(r"\s+", " ", title)
        link = entry.find("atom:id", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        summary = re.sub(r"\s+", " ", summary)
        published = entry.find("atom:published", ns).text.strip()[:10]

        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
        authors_str = ", ".join(authors[:3])
        if len(authors) > 3:
            authors_str += " et al."

        papers.append({
            "title": title,
            "link": link,
            "authors": authors_str,
            "date": published,
            "abstract": summary[:500],  # 截断避免过长
        })

    return papers


def fetch_all_arxiv():
    """抓取所有主题的 arXiv 论文"""
    all_results = {}
    for topic, keywords in ARXIV_TOPICS.items():
        print(f"正在抓取 arXiv: {topic} ...")
        seen_titles = set()
        papers = []
        for kw in keywords:
            results = fetch_arxiv_papers(kw)
            for p in results:
                if p["title"] not in seen_titles:
                    seen_titles.add(p["title"])
                    papers.append(p)
            time.sleep(1)  # 避免请求过快
        # 按日期排序
        papers.sort(key=lambda x: x["date"], reverse=True)
        all_results[topic] = papers[:15]  # 每个主题最多15篇
    return all_results


# ============================================================
# GitHub Trending 爬取
# ============================================================

def fetch_github_repos(query, max_results=5):
    """通过 GitHub Search API 搜索最近更新的仓库"""
    # 计算日期范围: 最近7天
    since = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    params = {
        "q": f"{query} pushed:>{since}",
        "sort": "stars",
        "order": "desc",
        "per_page": max_results,
    }
    url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "AutoResearchBot/1.0",
            "Accept": "application/vnd.github.v3+json",
        })
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [WARN] GitHub 请求失败 ({query}): {e}", file=sys.stderr)
        return []

    repos = []
    for item in data.get("items", []):
        repos.append({
            "name": item["full_name"],
            "url": item["html_url"],
            "stars": item["stargazers_count"],
            "description": (item.get("description") or "")[:200],
            "language": item.get("language", ""),
            "updated": item.get("pushed_at", "")[:10],
        })
    return repos


def fetch_all_github():
    """抓取所有主题的 GitHub 项目"""
    all_results = {}
    for topic, keywords in GITHUB_TOPICS.items():
        print(f"正在抓取 GitHub: {topic} ...")
        seen_names = set()
        repos = []
        for kw in keywords:
            results = fetch_github_repos(kw)
            for r in results:
                if r["name"] not in seen_names:
                    seen_names.add(r["name"])
                    repos.append(r)
            time.sleep(2)  # GitHub API 速率限制
        repos.sort(key=lambda x: x["stars"], reverse=True)
        all_results[topic] = repos[:10]
    return all_results


# ============================================================
# Google News 爬取
# ============================================================

def fetch_news(query, max_results=5):
    """从 Google News RSS 抓取新闻"""
    encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
            data = resp.read().decode("utf-8")
    except Exception as e:
        print(f"  [WARN] Google News 请求失败 ({query}): {e}", file=sys.stderr)
        return []

    root = ET.fromstring(data)
    items = root.findall(".//item")

    articles = []
    for item in items[:max_results]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()[:16]
        # Google News 描述字段包含 HTML，需要提取文本
        desc = (item.findtext("description") or "").strip()
        desc_clean = re.sub(r"<[^>]+>", "", desc)[:300]
        if title and link:
            articles.append({
                "title": title,
                "link": link,
                "date": pub_date,
                "description": desc_clean,
            })
    return articles


def fetch_all_news():
    """抓取所有主题的新闻"""
    all_results = {}
    for topic, keywords in NEWS_TOPICS.items():
        print(f"正在抓取新闻: {topic} ...")
        seen_titles = set()
        articles = []
        for kw in keywords:
            results = fetch_news(kw)
            for a in results:
                t = a["title"][:80]
                if t not in seen_titles:
                    seen_titles.add(t)
                    articles.append(a)
            time.sleep(1)
        all_results[topic] = articles[:8]
    return all_results


# ============================================================
# Gemini 摘要生成
# ============================================================

def build_prompt(arxiv_data, github_data, news_data):
    """构建 Gemini 的 Prompt"""
    today = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""你是一位自动驾驶与具身智能领域的研究分析师。请根据以下最新论文、新闻和开源项目，生成一份中文研究简报。

IMPORTANT - 输出格式要求：
1.  不要使用 #、##、### 等 markdown 标题语法。用加粗文字（**标题文字**）作为章节标题
2.  对每篇解读的论文，在其解读内容的最后一行附上链接，格式：🔗 [论文标题](http://arxiv.org/abs/xxxx)
3.  每条新闻点评后附上链接，格式：🔗 [新闻标题](url)
4.  每个主题挑选最有价值的 3-4 篇论文和 2-3 条重要新闻进行重点解读
5.  解读包括：核心创新点、技术路线、潜在影响
6.  对热门开源项目做简要点评
7.  最后给出"今日趋势总结"

---
数据日期: {today}

"""

    for topic, papers in arxiv_data.items():
        prompt += f"\n【{topic}】最新论文\n\n"
        for i, p in enumerate(papers, 1):
            prompt += f"{i}. {p['title']}\n"
            prompt += f"   作者: {p['authors']}\n"
            prompt += f"   日期: {p['date']}\n"
            prompt += f"   链接: {p['link']}\n"
            prompt += f"   摘要: {p['abstract']}\n\n"

    for topic, articles in news_data.items():
        prompt += f"\n【{topic}】业界新闻\n\n"
        for i, a in enumerate(articles, 1):
            prompt += f"{i}. {a['title']}\n"
            prompt += f"   日期: {a['date']}\n"
            prompt += f"   链接: {a['link']}\n"
            prompt += f"   摘要: {a['description']}\n\n"

    for topic, repos in github_data.items():
        prompt += f"\n【{topic}】热门开源项目\n\n"
        for i, r in enumerate(repos, 1):
            prompt += f"{i}. {r['name']} (⭐{r['stars']})\n"
            prompt += f"   语言: {r['language']}\n"
            prompt += f"   链接: {r['url']}\n"
            prompt += f"   描述: {r['description']}\n\n"

    return prompt


def call_gemini(prompt):
    """调用 Gemini API 生成摘要（优先官方 SDK，失败则回退到原生 HTTP + 重试）"""
    if not GEMINI_API_KEY:
        return "⚠️ 未配置 GEMINI_API_KEY，跳过摘要生成。原始数据已保存。"

    # --- 方案 1: 官方 SDK（优先，带内置重试）---
    # 先尝试新版 google.genai，其次兼容旧版 google.generativeai
    try:
        try:
            import google.genai as genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            model = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={"temperature": 0.3, "max_output_tokens": 8192},
            )
            text = model.text
            if text:
                return text
        except (ImportError, AttributeError):
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.3, "max_output_tokens": 8192},
                request_options={"timeout": 180},
            )
            text = response.text
            if text:
                return text
        return "⚠️ Gemini 返回为空 (SDK)"
    except ImportError:
        print("  google-generativeai 未安装，回退到 HTTP 调用...")
    except Exception as e:
        print(f"  SDK 调用失败，回退到 HTTP: {e}")

    # --- 方案 2: 原生 HTTP + 3 次指数退避重试 ---
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 8192,
        }
    }).encode("utf-8")

    last_err = None
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, data=payload, headers={
                "Content-Type": "application/json",
                "User-Agent": "DailyResearchBot/1.0",
            })
            with urllib.request.urlopen(req, timeout=180, context=_SSL_CTX) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts:
                        return parts[0].get("text", "⚠️ Gemini 返回为空")
                return f"⚠️ Gemini 返回异常: {json.dumps(data, ensure_ascii=False)[:300]}"
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                wait = 5 * attempt
                print(f"  HTTP {e.code}，{wait}秒后重试 (第{attempt}次)...")
                time.sleep(wait)
                last_err = f"API 错误 {e.code}"
                continue
            return f"⚠️ Gemini API 错误 ({e.code}): {body[:500]}"
        except Exception as e:
            last_err = str(e)
            if attempt < 3:
                wait = 5 * attempt
                print(f"  调用异常: {e}，{wait}秒后重试 (第{attempt}次)...")
                time.sleep(wait)
                continue
            return f"⚠️ Gemini 调用失败: {last_err}"

    return f"⚠️ Gemini 调用最终失败: {last_err}"


# ============================================================
# 飞书推送
# ============================================================

def send_to_feishu(report_text, arxiv_data, github_data, news_data):
    """通过飞书自定义机器人 Webhook 发送报告（只推送 AI 摘要卡片）"""
    if not FEISHU_WEBHOOK_URL:
        print("  未配置 FEISHU_WEBHOOK_URL，跳过飞书推送")
        return

    today = datetime.now().strftime("%Y-%m-%d")

    sections = []

    # AI 摘要（包含内嵌论文/新闻链接）
    sections.append({
        "tag": "markdown",
        "content": _feishu_md(report_text[:4500]),
    })

    # 热门开源项目（简短）
    gh_lines = []
    count = 0
    for topic, repos in github_data.items():
        for r in repos[:2]:
            count += 1
            if count > 6:
                break
            gh_lines.append(f"{count}. [{r['name']}]({r['url']}) ⭐{r['stars']}")
        if count > 6:
            break

    if gh_lines:
        sections.append({"tag": "hr"})
        sections.append({"tag": "markdown", "content": "**热门开源项目**\n" + "\n".join(gh_lines)})

    card = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": f"📰 每日研究简报 {today}"},
                "template": "blue",
            },
            "elements": sections,
        },
    }

    _post_feishu(card, "每日简报")


def _feishu_md(text):
    """将标准 Markdown 转换为飞书卡片兼容的 Markdown"""
    # 飞书卡片不支持 # 标题语法，移除行首 #+ 前缀
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # 飞书卡片不支持 HTML 标签
    text = re.sub(r"</?details[^>]*>", "", text)
    text = re.sub(r"</?summary[^>]*>", "**", text)
    text = re.sub(r"<hr\s*/?>", "---", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text


def _post_feishu(payload, label=""):
    """发送一条飞书消息"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        FEISHU_WEBHOOK_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if result.get("code") != 0:
                print(f"  [WARN] 飞书推送失败 ({label}): {result}")
            else:
                print(f"  飞书推送成功: {label}")
    except Exception as e:
        print(f"  [WARN] 飞书推送异常 ({label}): {e}")


# ============================================================
# 报告输出
# ============================================================

def save_report(report_text, arxiv_data, github_data, news_data):
    """保存报告为 Markdown 文件"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    filepath = os.path.join(REPORT_DIR, f"{today}.md")

    # 组装完整报告
    full_report = f"# 自动驾驶 & 具身智能 每日研究简报\n\n"
    full_report += f"**日期**: {today}\n\n---\n\n"
    full_report += report_text
    full_report += "\n\n---\n\n"
    full_report += "<details>\n<summary>📋 原始数据（点击展开）</summary>\n\n"

    for topic, papers in arxiv_data.items():
        full_report += f"\n### {topic} 论文\n\n"
        for p in papers:
            full_report += f"- [{p['title']}]({p['link']}) ({p['date']})\n"

    for topic, articles in news_data.items():
        full_report += f"\n### {topic} 新闻\n\n"
        for a in articles:
            full_report += f"- [{a['title']}]({a['link']})\n"

    for topic, repos in github_data.items():
        full_report += f"\n### {topic} - GitHub\n\n"
        for r in repos:
            full_report += f"- [{r['name']}]({r['url']}) ⭐{r['stars']}\n"

    full_report += "\n</details>\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"\n报告已保存: {filepath}")
    return filepath


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("自动驾驶 VLA / 世界模型 / 具身智能 每日研究简报")
    print("=" * 60)

    # 1. 抓取 arXiv
    print("\n[1/5] 抓取 arXiv 最新论文 ...")
    arxiv_data = fetch_all_arxiv()
    total_papers = sum(len(v) for v in arxiv_data.values())
    print(f"  共获取 {total_papers} 篇论文")

    # 2. 抓取 GitHub
    print("\n[2/5] 抓取 GitHub 热门项目 ...")
    github_data = fetch_all_github()
    total_repos = sum(len(v) for v in github_data.values())
    print(f"  共获取 {total_repos} 个项目")

    # 3. 抓取新闻
    print("\n[3/5] 抓取业界新闻 ...")
    news_data = fetch_all_news()
    total_news = sum(len(v) for v in news_data.values())
    print(f"  共获取 {total_news} 条新闻")

    # 4. 生成摘要
    print("\n[4/5] 生成 AI 摘要 ...")
    prompt = build_prompt(arxiv_data, github_data, news_data)
    report_text = call_gemini(prompt)
    print(f"  摘要长度: {len(report_text)} 字符")

    # 5. 保存 + 推送
    filepath = save_report(report_text, arxiv_data, github_data, news_data)

    print("\n[5/5] 推送到飞书 ...")
    send_to_feishu(report_text, arxiv_data, github_data, news_data)

    # 输出预览
    print("\n" + "=" * 60)
    print("报告预览 (前 2000 字符):")
    print("=" * 60)
    print(report_text[:2000])

    return filepath


if __name__ == "__main__":
    main()
