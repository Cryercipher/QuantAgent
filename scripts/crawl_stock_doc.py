import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import urljoin, urlparse

# Configuration
START_URL = "https://hellowac.github.io/stock_doc/stock/base/code_prefix/"
BASE_DOMAIN = "https://hellowac.github.io/stock_doc/"
OUTPUT_FILE = "stock_doc_content.txt"

def get_soup(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response.encoding = 'utf-8'
        return BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_content(soup):
    """Extracts the main content from the MkDocs Material page."""
    if not soup:
        return None, None

    # MkDocs Material usually puts content in article.md-content__inner
    article = soup.find('article', class_='md-content__inner')
    
    # If not found, try a more generic approach or other common structures
    if not article:
        article = soup.find('div', role='main')
    
    if not article:
        # Fallback: try to find the main content by excluding nav and header
        # This is a rough fallback
        body = soup.find('body')
        if body:
            article = body
    
    if not article:
        return None, None

    # Extract Title
    title_tag = article.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else "No Title"

    # Clean up content
    # Remove script and style tags
    for script in article(["script", "style"]):
        script.decompose()

    # Insert newlines around block elements to preserve structure
    for tag in article.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'ul', 'ol', 'blockquote', 'pre', 'table', 'br']):
        tag.insert_before('\n')
        tag.insert_after('\n')

    # Get text without separator so inline elements stay together
    text = article.get_text()
    
    # Basic cleaning
    lines = [line.strip() for line in text.split('\n')]
    # Remove empty lines
    clean_lines = [line for line in lines if line]
    clean_text = '\n'.join(clean_lines)
    
    return title, clean_text

def get_all_links(soup, base_url):
    """Extracts all links from the sidebar navigation."""
    links = []
    if not soup:
        return links
    
    # MkDocs Material sidebar navigation
    nav = soup.find('nav', class_='md-nav')
    if nav:
        for a in nav.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(base_url, href)
            
            # Filter links to keep only those within the documentation scope
            if full_url.startswith(BASE_DOMAIN):
                # Remove anchors
                full_url = full_url.split('#')[0]
                if full_url not in links:
                    links.append(full_url)
    
    return links

def main():
    print(f"Starting crawl from {START_URL}...")
    
    # 1. Get the start page to extract navigation links
    soup = get_soup(START_URL)
    if not soup:
        print("Failed to retrieve start page.")
        return

    # 2. Extract all unique links from the navigation sidebar to ensure order
    # Using a list to preserve order found in the sidebar
    all_urls = get_all_links(soup, START_URL)
    
    # Ensure the start URL is included if not found (it should be)
    if START_URL not in all_urls:
        # Check if a normalized version is in there
        if START_URL.rstrip('/') not in [u.rstrip('/') for u in all_urls]:
             all_urls.insert(0, START_URL)
    
    print(f"Found {len(all_urls)} pages to crawl.")
    
    # 3. Crawl each page
    results = []
    visited = set()
    
    for i, url in enumerate(all_urls):
        if url in visited:
            continue
        visited.add(url)
        
        print(f"[{i+1}/{len(all_urls)}] Crawling: {url}")
        
        page_soup = get_soup(url)
        if not page_soup:
            continue
            
        title, content = extract_content(page_soup)
        
        if title and content:
            page_data = f"xx_START_PAGE_xx\nURL: {url}\nTITLE: {title}\n\n{content}\nxx_END_PAGE_xx\n"
            results.append(page_data)
        
        # Be polite
        time.sleep(0.5)

    # 4. Save to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    
    print(f"Done! Saved content to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
