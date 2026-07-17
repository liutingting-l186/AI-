import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.crta.org.cn"
LIST_URL = "https://www.crta.org.cn/news.html?id=12"
OUTPUT_DIR = "crta_industry_news"
STATE_FILE = "spider_state.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed_ids": [], "last_page": 1}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def sanitize_filename(filename):
    invalid_chars = r'[\\/:*?"<>|\r\n]'
    return re.sub(invalid_chars, "_", filename)


def fetch_page(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        response.encoding = "utf-8"
        return response.text
    except Exception as e:
        print(f"❌ 请求失败: {url}, 错误: {e}")
        return None


def parse_list_page(html):
    soup = BeautifulSoup(html, "html.parser")
    articles = []
    seen_ids = set()
    
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "details.html?id=12&contentId=" in href:
            match = re.search(r"contentId=(\d+)", href)
            if match:
                content_id = match.group(1)
                if content_id in seen_ids:
                    continue
                title = link.get_text(strip=True)
                if not title:
                    continue
                seen_ids.add(content_id)
                if href.startswith("//"):
                    url = "https:" + href
                elif href.startswith("/"):
                    url = BASE_URL + href
                elif href.startswith("http"):
                    url = href
                else:
                    url = BASE_URL + "/" + href
                articles.append({"content_id": content_id, "title": title, "url": url})
    
    page_links = soup.find_all("a", href=True)
    max_page = 1
    for link in page_links:
        href = link["href"]
        if "pageIndex" in href and "id=12" in href:
            match = re.search(r"pageIndex=(\d+)", href)
            if match:
                max_page = max(max_page, int(match.group(1)))
    
    return articles, max_page


def parse_detail_page(html):
    soup = BeautifulSoup(html, "html.parser")
    
    title = ""
    title_tag = soup.find("h1") or soup.find("h2") or soup.find("h3")
    if title_tag:
        title = title_tag.get_text(strip=True)
    
    author = ""
    time_str = ""
    info_tags = soup.find_all("span")
    for tag in info_tags:
        text = tag.get_text(strip=True)
        if "作者" in text and not author:
            author = text
        elif "时间" in text and not time_str:
            time_str = text
    
    content_div = soup.find("div", class_="cenRightContent")
    if content_div:
        paragraphs = content_div.find_all("p")
        valid_paragraphs = []
        
        for i, p in enumerate(paragraphs):
            text = p.get_text(strip=True)
            if not text or len(text) < 5:
                continue
            
            if i == 0 and "作者" in text and "时间" in text:
                continue
            
            is_compressed = False
            for j in range(i + 1, len(paragraphs)):
                next_text = paragraphs[j].get_text(strip=True)
                if next_text and len(next_text) > 10 and next_text in text:
                    is_compressed = True
                    break
            
            if not is_compressed:
                valid_paragraphs.append(text)
        
        lines = valid_paragraphs
        
        content = "\n\n".join(lines)
    else:
        content = ""
    
    return {
        "title": title,
        "author": author,
        "time": time_str,
        "content": content.strip()
    }


def save_article(article):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    date_str = ""
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", article["time"])
    if date_match:
        date_str = date_match.group(1)
    else:
        date_str = "unknown"
    
    filename = f"{date_str}_{sanitize_filename(article['title'])}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    content = f"【标题】{article['title']}\n"
    content += f"【作者】{article['author']}\n"
    content += f"【时间】{article['time']}\n"
    content += f"【链接】{article['url']}\n"
    content += "=" * 50 + "\n\n"
    content += article["content"]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    return filepath


def crawl():
    state = load_state()
    processed_ids = set(state.get("processed_ids", []))
    last_page = state.get("last_page", 1)
    
    print(f"🚀 开始抓取行业资讯...")
    print(f"📋 已处理文章数: {len(processed_ids)}")
    print(f"📄 上次抓取到第 {last_page} 页")
    
    current_page = 1
    max_page = 1
    new_count = 0
    skipped_count = 0
    
    while current_page <= max_page:
        url = f"{LIST_URL}&pageIndex={current_page}"
        
        print(f"\n📖 正在抓取第 {current_page} 页: {url}")
        html = fetch_page(url)
        if not html:
            time.sleep(3)
            continue
        
        articles, page_max = parse_list_page(html)
        max_page = max(max_page, page_max)
        
        print(f"📝 本页共 {len(articles)} 篇文章")
        
        for article in articles:
            content_id = article["content_id"]
            
            if content_id in processed_ids:
                print(f"⏭️ 跳过已处理: {article['title']}")
                skipped_count += 1
                continue
            
            print(f"🔍 正在抓取: {article['title']}")
            detail_html = fetch_page(article["url"])
            if not detail_html:
                print(f"❌ 抓取详情失败: {article['title']}")
                time.sleep(2)
                continue
            
            detail = parse_detail_page(detail_html)
            article.update(detail)
            
            filepath = save_article(article)
            print(f"✅ 已保存: {filepath}")
            
            processed_ids.add(content_id)
            new_count += 1
            
            time.sleep(1)
        
        current_page += 1
        time.sleep(2)
    
    state["processed_ids"] = list(processed_ids)
    state["last_page"] = max_page
    state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)
    
    print(f"\n🎉 抓取完成！")
    print(f"📊 新增文章: {new_count} 篇")
    print(f"⏭️ 跳过已处理: {skipped_count} 篇")
    print(f"📁 总计保存: {len(processed_ids)} 篇")
    print(f"📂 保存目录: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    crawl()