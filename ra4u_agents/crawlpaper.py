from serpapi import GoogleSearch
import json
from datetime import datetime
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import urllib.parse
from pathlib import Path
import PyPDF2
import io

@dataclass
class Paper:
    """Class để represent một paper từ Google Scholar"""
    title: str
    authors: str
    year: int
    venue: str
    citations: int
    snippet: str
    link: str
    paper_content: str = ""
    pdf_url: str = ""
    pdf_path: str = ""
    pdf_content: str = ""
    relevance_score: float = 0.0
    venue_score: float = 0.0
    recency_score: float = 0.0
    total_score: float = 0.0

class ScholarCrawler:
    """Class để crawl papers từ Google Scholar API của SerpApi"""
    
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            self.api_key = api_key
        else:
            load_dotenv()
            self.api_key = os.getenv('SERPAPI_API_KEY')
            if not self.api_key:
                raise ValueError("API key không tìm thấy. Vui lòng set SERPAPI_API_KEY trong file .env hoặc truyền vào constructor.")
        
        # Danh sách các venue/conference uy tín theo từng lĩnh vực
        self.prestigious_venues = {
            'computer_science': [
                'ICML', 'NeurIPS', 'ICLR', 'AAAI', 'IJCAI', 'CVPR', 'ICCV', 'ECCV',
                'SIGIR', 'WWW', 'SIGKDD', 'SIGMOD', 'VLDB', 'ICDE', 'OSDI', 'SOSP',
                'PLDI', 'POPL', 'OOPSLA', 'ISCA', 'MICRO', 'ASPLOS', 'CHI', 'UIST',
                'ICSE', 'FSE', 'ASE', 'ISSTA', 'CCS', 'NDSS', 'USENIX Security',
                'S&P', 'CRYPTO', 'EUROCRYPT', 'ASIACRYPT', 'TCC', 'STOC', 'FOCS',
                'SODA', 'ICALP', 'ESA', 'SPAA', 'PODC', 'DISC'
            ],
            'biology': [
                'Nature', 'Science', 'Cell', 'PNAS', 'Nature Biotechnology',
                'Nature Medicine', 'Nature Genetics', 'Nature Neuroscience',
                'Nature Immunology', 'Nature Cell Biology', 'Molecular Cell',
                'Developmental Cell', 'Current Biology', 'EMBO Journal',
                'Journal of Cell Biology', 'PLoS Biology', 'eLife'
            ],
            'physics': [
                'Physical Review Letters', 'Nature Physics', 'Science',
                'Physical Review A', 'Physical Review B', 'Physical Review C',
                'Physical Review D', 'Physical Review E', 'Reviews of Modern Physics',
                'Annual Review of Nuclear and Particle Science', 'Physics Reports'
            ],
            'chemistry': [
                'Journal of the American Chemical Society', 'Angewandte Chemie',
                'Chemical Reviews', 'Nature Chemistry', 'Chemical Science',
                'Accounts of Chemical Research', 'Chemical Communications',
                'Organic Letters', 'Inorganic Chemistry', 'Analytical Chemistry'
            ]
        }
    
    def search_papers(self, query: str, num_results: int = 50) -> List[Paper]:
        """
        Tìm kiếm papers từ Google Scholar API
        
        Args:
            query: Research query để tìm kiếm
            num_results: Số lượng results muốn lấy
            
        Returns:
            List các Paper objects
        """
        papers = []
        start = 0
        results_per_page = 10
        
        while len(papers) < num_results:
            params = {
                "engine": "google_scholar",
                "q": query,
                "api_key": self.api_key,
                "start": start,
                "num": min(results_per_page, num_results - len(papers))
            }
            
            try:
                search = GoogleSearch(params)
                data = search.get_dict()
                
                if "organic_results" not in data:
                    print(f"Không tìm thấy results cho query: {query}")
                    break
                
                for result in data["organic_results"]:
                    paper = self._parse_paper(result)
                    if paper:
                        papers.append(paper)
                
                # Kiểm tra xem còn results nữa không
                if len(data["organic_results"]) < results_per_page:
                    break
                    
                start += results_per_page
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Lỗi khi gọi API: {e}")
                break
        
        return papers[:num_results]
    
    def find_pdf_url(self, paper: Paper) -> str:
        """
        Tìm PDF URL từ paper link
        
        Args:
            paper: Paper object
            
        Returns:
            PDF URL nếu tìm thấy, empty string nếu không
        """
        if not paper.link:
            return ""
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Kiểm tra nếu link đã là PDF
            if paper.link.lower().endswith('.pdf'):
                return paper.link
            
            # Xử lý arXiv links
            if 'arxiv.org' in paper.link:
                # Convert arXiv abstract URL to PDF URL
                if '/abs/' in paper.link:
                    pdf_url = paper.link.replace('/abs/', '/pdf/') + '.pdf'
                    return pdf_url
                elif '/pdf/' in paper.link:
                    return paper.link
            
            # Lấy nội dung trang để tìm PDF links
            response = requests.get(paper.link, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Tìm các PDF links phổ biến
            pdf_patterns = [
                # Direct PDF links
                'a[href$=".pdf"]',
                'a[href*=".pdf"]',
                # arXiv PDF links
                'a[href*="arxiv.org/pdf/"]',
                # ResearchGate PDF links
                'a[href*="researchgate.net"][href*=".pdf"]',
                # IEEE Xplore
                'a[href*="ieeexplore.ieee.org"][href*="stamp"]',
                # ACM Digital Library
                'a[href*="dl.acm.org"][href*=".pdf"]',
                # Springer
                'a[href*="link.springer.com"][href*=".pdf"]',
                # Generic PDF download buttons/links
                'a:contains("PDF")',
                'a:contains("Download PDF")',
                'a:contains("Full Text PDF")',
                'button:contains("PDF")'
            ]
            
            for pattern in pdf_patterns:
                elements = soup.select(pattern)
                for element in elements:
                    href = element.get('href', '')
                    if href:
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            base_url = urllib.parse.urljoin(paper.link, '/')
                            href = urllib.parse.urljoin(base_url, href)
                        elif not href.startswith('http'):
                            href = urllib.parse.urljoin(paper.link, href)
                        
                        # Validate PDF URL
                        if self._is_valid_pdf_url(href):
                            return href
            
            # Tìm trong meta tags
            meta_pdf = soup.find('meta', {'name': 'citation_pdf_url'})
            if meta_pdf and meta_pdf.get('content'):
                return meta_pdf['content']
            
            return ""
            
        except Exception as e:
            print(f"Lỗi khi tìm PDF cho paper: {paper.title} - {e}")
            return ""
    
    def _is_valid_pdf_url(self, url: str) -> bool:
        """Kiểm tra URL có phải là PDF hợp lệ không"""
        try:
            # Basic URL validation
            if not url.startswith('http'):
                return False
            
            # Check if URL ends with .pdf or contains PDF indicators
            if url.lower().endswith('.pdf') or 'pdf' in url.lower():
                return True
            
            return False
        except:
            return False
    
    def download_pdf(self, paper: Paper, download_dir: str = "papers") -> bool:
        """
        Tải PDF của paper
        
        Args:
            paper: Paper object
            download_dir: Thư mục lưu PDF
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        if not paper.pdf_url:
            paper.pdf_url = self.find_pdf_url(paper)
        
        if not paper.pdf_url:
            print(f"Không tìm thấy PDF cho paper: {paper.title}")
            return False
        
        try:
            # Tạo thư mục nếu chưa có
            Path(download_dir).mkdir(exist_ok=True)
            
            # Tạo tên file an toàn
            safe_title = re.sub(r'[<>:"/\\|?*]', '_', paper.title)[:100]
            filename = f"{safe_title}_{paper.year}.pdf"
            filepath = Path(download_dir) / filename
            
            print(f"Đang tải PDF: {paper.title}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(paper.pdf_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Kiểm tra content type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not paper.pdf_url.lower().endswith('.pdf'):
                print(f"URL không trả về PDF: {paper.pdf_url}")
                return False
            
            # Lưu file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            paper.pdf_path = str(filepath)
            print(f"✅ Đã tải PDF: {filepath}")
            
            # Đọc nội dung PDF
            paper.pdf_content = self.extract_pdf_text(str(filepath))
            
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi tải PDF cho paper {paper.title}: {e}")
            return False
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """
        Trích xuất text từ PDF file
        
        Args:
            pdf_path: Đường dẫn đến PDF file
            
        Returns:
            Text content của PDF
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            # Làm sạch text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Giới hạn độ dài
            return text[:50000] if len(text) > 50000 else text
            
        except Exception as e:
            print(f"Lỗi khi đọc PDF {pdf_path}: {e}")
            return ""
    
    def get_paper_content(self, paper: Paper) -> str:
        """
        Lấy nội dung paper từ link (nếu có thể truy cập)
        
        Args:
            paper: Paper object
            
        Returns:
            Nội dung paper hoặc empty string nếu không lấy được
        """
        if not paper.link:
            return ""
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(paper.link, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Xóa script và style tags
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Lấy text content
            text = soup.get_text()
            
            # Làm sạch text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Giới hạn độ dài (10000 characters)
            return text[:10000] if len(text) > 10000 else text
            
        except Exception as e:
            print(f"Không thể lấy nội dung cho paper: {paper.title} - Lỗi: {e}")
            return ""
    
    def _parse_paper(self, result: Dict[str, Any]) -> Paper:
        """Parse một result từ API thành Paper object"""
        try:
            title = result.get("title", "")
            
            # Parse publication info
            pub_info = result.get("publication_info", {})
            summary = pub_info.get("summary", "")
            
            # Extract year từ summary
            year_match = re.search(r'\b(19|20)\d{2}\b', summary)
            year = int(year_match.group()) if year_match else datetime.now().year
            
            # Extract authors và venue
            authors = ""
            venue = ""
            if summary:
                parts = summary.split(" - ")
                if len(parts) >= 2:
                    authors = parts[0]
                    venue = parts[-1] if len(parts) > 2 else parts[1]
            
            # Get citations
            citations = 0
            inline_links = result.get("inline_links", {})
            cited_by = inline_links.get("cited_by", {})
            if cited_by:
                citations = cited_by.get("total", 0)
            
            snippet = result.get("snippet", "")
            link = result.get("link", "")
            
            return Paper(
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                citations=citations,
                snippet=snippet,
                link=link
            )
            
        except Exception as e:
            print(f"Lỗi khi parse paper: {e}")
            return None
    
    def calculate_relevance_score(self, paper: Paper, query: str) -> float:
        """
        Tính điểm độ liên quan dựa trên title và snippet
        
        Args:
            paper: Paper object
            query: Original search query
            
        Returns:
            Relevance score từ 0-1
        """
        query_terms = query.lower().split()
        text = (paper.title + " " + paper.snippet).lower()
        
        # Đếm số terms xuất hiện
        matches = sum(1 for term in query_terms if term in text)
        base_score = matches / len(query_terms)
        
        # Bonus cho exact phrase matches
        if query.lower() in text:
            base_score += 0.2
        
        # Normalize citations (log scale)
        citation_bonus = min(0.3, paper.citations / 1000) if paper.citations > 0 else 0
        
        return min(1.0, base_score + citation_bonus)
    
    def calculate_venue_score(self, paper: Paper, field: str = 'computer_science') -> float:
        """
        Tính điểm venue dựa trên danh sách prestigious venues
        
        Args:
            paper: Paper object
            field: Research field để chọn venue list phù hợp
            
        Returns:
            Venue score từ 0-1
        """
        if field not in self.prestigious_venues:
            field = 'computer_science'  # Default
        
        prestigious_list = self.prestigious_venues[field]
        venue_lower = paper.venue.lower()
        
        for venue in prestigious_list:
            if venue.lower() in venue_lower:
                return 1.0
        
        # Partial matching cho journal names
        for venue in prestigious_list:
            venue_words = venue.lower().split()
            if any(word in venue_lower for word in venue_words if len(word) > 3):
                return 0.7
        
        # Bonus cho high citation venues
        if paper.citations > 100:
            return 0.5
        elif paper.citations > 50:
            return 0.3
        
        return 0.1
    
    def calculate_recency_score(self, paper: Paper) -> float:
        """
        Tính điểm độ gần đây
        
        Args:
            paper: Paper object
            
        Returns:
            Recency score từ 0-1
        """
        current_year = datetime.now().year
        years_diff = current_year - paper.year
        
        if years_diff <= 1:
            return 1.0
        elif years_diff <= 3:
            return 0.8
        elif years_diff <= 5:
            return 0.6
        elif years_diff <= 10:
            return 0.4
        else:
            return 0.2
    
    def rerank_papers(self, papers: List[Paper], query: str, field: str = 'computer_science',
                     weights: Dict[str, float] = None) -> List[Paper]:
        """
        Rerank papers theo các tiêu chí
        
        Args:
            papers: List các Paper objects
            query: Original search query
            field: Research field
            weights: Trọng số cho từng tiêu chí
            
        Returns:
            List papers đã được rerank
        """
        if weights is None:
            weights = {
                'relevance': 0.4,
                'venue': 0.35,
                'recency': 0.25
            }
        
        # Tính scores cho từng paper
        for paper in papers:
            paper.relevance_score = self.calculate_relevance_score(paper, query)
            paper.venue_score = self.calculate_venue_score(paper, field)
            paper.recency_score = self.calculate_recency_score(paper)
            
            # Tính total score
            paper.total_score = (
                paper.relevance_score * weights['relevance'] +
                paper.venue_score * weights['venue'] +
                paper.recency_score * weights['recency']
            )
        
        # Sort theo total score
        return sorted(papers, key=lambda p: p.total_score, reverse=True)
    
    def search_and_rank(self, query: str, field: str = 'computer_science', 
                       num_results: int = 50, weights: Dict[str, float] = None, 
                       fetch_content: bool = True) -> List[Paper]:
        """
        Main function: Search và rerank papers
        
        Args:
            query: Research query
            field: Research field
            num_results: Số lượng results
            weights: Custom weights
            fetch_content: Có lấy nội dung paper không
            
        Returns:
            List papers đã được ranked
        """
        print(f"Đang tìm kiếm papers cho query: '{query}'...")
        papers = self.search_papers(query, num_results)
        
        
        
        print(f"Tìm thấy {len(papers)} papers. Đang rerank...")
        ranked_papers = self.rerank_papers(papers, query, field, weights)
        
        return ranked_papers
    
    def download_top_papers(self, papers: List[Paper], top_k: int = 5, download_dir: str = "papers") -> List[Paper]:
        """
        Tải PDF của top K papers
        
        Args:
            papers: List papers đã được ranked
            top_k: Số lượng papers cần tải
            download_dir: Thư mục lưu PDF
            
        Returns:
            List papers với PDF đã tải
        """
        top_papers = papers[:top_k]
        successful_downloads = []
        
        print(f"\n🔽 ĐANG TẢI TOP {top_k} PAPERS...")
        print("=" * 50)
        
        for i, paper in enumerate(top_papers, 1):
            print(f"\n[{i}/{top_k}] {paper.title}")
            
            # Tìm PDF URL
            print("🔍 Đang tìm PDF link...")
            paper.pdf_url = self.find_pdf_url(paper)
            
            if paper.pdf_url:
                print(f"✅ Tìm thấy PDF: {paper.pdf_url}")
                
                # Tải PDF
                success = self.download_pdf(paper, download_dir)
                if success:
                    successful_downloads.append(paper)
                    print(f"📄 PDF Content Length: {len(paper.pdf_content)} characters")
                else:
                    print("❌ Không thể tải PDF")
            else:
                print("❌ Không tìm thấy PDF link")
            
            print("-" * 30)
            time.sleep(2)  # Rate limiting
        
        print(f"\n✅ Đã tải thành công {len(successful_downloads)}/{top_k} papers")
        return successful_downloads
    
    def print_results(self, papers: List[Paper], top_k: int = 10):
        """In kết quả top papers"""
        print(f"\n=== TOP {min(top_k, len(papers))} PAPERS ===\n")
        
        for i, paper in enumerate(papers[:top_k], 1):
            print(f"{i}. {paper.title}")
            print(f"   Authors: {paper.authors}")
            print(f"   Venue: {paper.venue} ({paper.year})")
            print(f"   Citations: {paper.citations}")
            print(f"   Content Length: {len(paper.paper_content)} characters")
            if paper.pdf_url:
                print(f"   📄 PDF URL: {paper.pdf_url}")
            if paper.pdf_path:
                print(f"   💾 PDF Downloaded: {paper.pdf_path}")
                print(f"   📖 PDF Content Length: {len(paper.pdf_content)} characters")
            print(f"   Scores - Relevance: {paper.relevance_score:.3f}, "
                  f"Venue: {paper.venue_score:.3f}, "
                  f"Recency: {paper.recency_score:.3f}")
            print(f"   Total Score: {paper.total_score:.3f}")
            print(f"   Link: {paper.link}")
            if paper.pdf_content:
                print(f"   PDF Preview: {paper.pdf_content[:200]}...")
            elif paper.paper_content:
                print(f"   Content Preview: {paper.paper_content[:200]}...")
            print("-" * 80)

def main():
    """Demo function"""
    # Sử dụng API key từ .env file
    crawler = ScholarCrawler()
    
    # Example usage
    query = "machine learning natural language processing"
    field = "computer_science"
    
    # Custom weights (optional)
    custom_weights = {
        'relevance': 0.5,    # Tăng trọng số độ liên quan
        'venue': 0.3,       # Giảm trọng số venue
        'recency': 0.2      # Giảm trọng số thời gian
    }
    
    try:
        # Tìm kiếm và rank papers (không lấy content để nhanh hơn)
        papers = crawler.search_and_rank(
            query=query,
            field=field,
            num_results=30,
            weights=custom_weights,
            fetch_content=False  # Không lấy content trang web để tập trung vào PDF
        )
        
        # Hiển thị kết quả ban đầu
        crawler.print_results(papers, top_k=10)
        
        # Tải PDF cho top 5 papers
        downloaded_papers = crawler.download_top_papers(papers, top_k=5, download_dir="papers")
        
        # Hiển thị kết quả sau khi tải PDF
        if downloaded_papers:
            print(f"\n📚 KẾT QUẢ SAU KHI TẢI PDF:")
            crawler.print_results(downloaded_papers, top_k=len(downloaded_papers))
        
        # Export to JSON với thông tin PDF
        export_data = []
        for paper in papers[:10]:
            export_data.append({
                'title': paper.title,
                'authors': paper.authors,
                'venue': paper.venue,
                'year': paper.year,
                'citations': paper.citations,
                'snippet': paper.snippet,
                'paper_content': paper.paper_content,
                'pdf_url': paper.pdf_url,
                'pdf_path': paper.pdf_path,
                'pdf_content': paper.pdf_content[:1000] if paper.pdf_content else "",  # Chỉ lấy 1000 ký tự đầu
                'total_score': paper.total_score,
                'relevance_score': paper.relevance_score,
                'venue_score': paper.venue_score,
                'recency_score': paper.recency_score,
                'link': paper.link
            })
        
        with open('ranked_papers_with_pdf.json', 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Đã export {len(export_data)} papers vào file 'ranked_papers_with_pdf.json'")
        
        # Thống kê
        pdf_count = sum(1 for p in papers[:10] if p.pdf_path)
        print(f"\n📊 THỐNG KÊ:")
        print(f"   • Tổng papers tìm thấy: {len(papers)}")
        print(f"   • Papers có PDF: {pdf_count}/10")
        print(f"   • Thư mục lưu PDF: ./papers/")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()
