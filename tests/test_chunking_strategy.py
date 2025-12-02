import re
import sys
from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# 配置路径
FILE_PATH = "/root/nas-private/QuantAgent/data/rules_text/stock_doc_content.txt"

def main():
    print(f"[*] 开始读取文件: {FILE_PATH}")
    try:
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"[!] 文件未找到: {FILE_PATH}")
        return

    print(f"[*] 文件读取成功，总字符数: {len(raw_text)}")

    # --- 第一级切分：逻辑文档拆分 ---
    print("\n[1] 执行第一级切分 (Logical Document Splitting)...")
    
    # 匹配 xx_START_PAGE_xx ... xx_END_PAGE_xx 之间的内容
    # 使用非贪婪匹配 (.*?) 和 DOTALL 模式
    page_pattern = re.compile(r"xx_START_PAGE_xx\n(.*?)\nxx_END_PAGE_xx", re.DOTALL)
    matches = page_pattern.findall(raw_text)
    
    print(f"[*] 识别到逻辑页面数量: {len(matches)}")

    documents: List[Document] = []
    
    for i, page_content in enumerate(matches):
        lines = page_content.strip().split("\n")
        metadata = {"source_idx": i}
        content_lines = []
        
        # 提取元数据 & 清洗正文
        for line in lines:
            # 简单的行首匹配提取元数据
            if line.startswith("URL:"):
                metadata["url"] = line.replace("URL:", "").strip()
            elif line.startswith("TITLE:"):
                metadata["title"] = line.replace("TITLE:", "").strip()
            else:
                content_lines.append(line)
        
        text = "\n".join(content_lines).strip()
        
        if not text:
            print(f"[!] 警告: 第 {i} 页内容为空，跳过。Metadata: {metadata}")
            continue
            
        doc = Document(text=text, metadata=metadata)
        documents.append(doc)

    print(f"[*] 成功构建 Document 对象数量: {len(documents)}")
    
    # 打印前 3 个文档的样本进行检查
    print("\n--- [Debug] 前 3 个逻辑文档样本 ---")
    for i, doc in enumerate(documents[:3]):
        print(f"Doc #{i}:")
        print(f"  Metadata: {doc.metadata}")
        preview = doc.text[:100].replace('\n', '\\n') + "..."
        print(f"  Content Preview: {preview}")
        print("-" * 40)

    # --- 第二级切分：语义文本切分 ---
    print("\n[2] 执行第二级切分 (Semantic Text Splitting)...")
    
    # 使用 SentenceSplitter，参数与之前项目设置保持一致或根据建议调整
    # chunk_size=512, chunk_overlap=64 (推荐值)
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    
    nodes = splitter.get_nodes_from_documents(documents)
    
    print(f"[*] 切分完成，生成 Nodes (Chunks) 总数: {len(nodes)}")
    
    # 打印前 5 个 Node 的样本进行检查
    print("\n--- [Debug] 前 5 个 Chunk 样本 ---")
    for i, node in enumerate(nodes[:5]):
        print(f"Chunk #{i}:")
        print(f"  Source Doc Title: {node.metadata.get('title', 'N/A')}")
        print(f"  Source URL: {node.metadata.get('url', 'N/A')}")
        print(f"  Length (chars): {len(node.text)}")
        print(f"  Text:\n{node.text}")
        print("-" * 40)

    # 检查是否有元数据丢失的情况
    missing_meta_count = sum(1 for n in nodes if 'title' not in n.metadata or 'url' not in n.metadata)
    if missing_meta_count > 0:
        print(f"\n[!] 警告: 有 {missing_meta_count} 个 Chunk 缺失 title 或 url 元数据。")
    else:
        print("\n[*] 所有 Chunk 均包含完整元数据。")

if __name__ == "__main__":
    main()
