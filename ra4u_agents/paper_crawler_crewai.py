"""
Paper Crawler Agent for CrewAI - All-in-One Implementation
Tích hợp hoàn chỉnh paper crawling tool với CrewAI framework

Author: AgenticRS Team
Version: 1.0.0
Dependencies: crewai, serpapi, requests, beautifulsoup4, PyPDF2, python-dotenv

Usage:
    from paper_crawler_crewai import PaperCrawlerFactory
    
    # Tạo agent cho Manager
    crawler = PaperCrawlerFactory.create_agent()
    agent = crawler.get_agent()  # CrewAI Agent instance
    
    # Sử dụng
    result = crawler.search_papers("machine learning NLP")
"""

import os
import json
import time
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import urllib.parse

# Core dependencies
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import PyPDF2
from serpapi import GoogleSearch

# CrewAI dependencies
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ================================
# LLM CONFIGURATION
# ================================

# Load environment variables
load_dotenv()

# Initialize Gemini LLM client
LLM_CLIENT = LLM(model="gemini/gemini-2.0-flash")


# ================================
# CORE DATA STRUCTURES
# ================================

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


# ================================
# SCHOLAR CRAWLER CORE
# ================================

class ScholarCrawler:
    """Core crawler logic cho Google Scholar API"""
    
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            self.api_key = api_key
        else:
            load_dotenv()
            self.api_key = os.getenv('SERPAPI_API_KEY')
            if not self.api_key:
                raise ValueError("API key không tìm thấy. Vui lòng set SERPAPI_API_KEY trong file .env")
        
        # Danh sách venues uy tín theo từng lĩnh vực
        self.prestigious_venues = {
            'computer_science': [
                'ICML', 'NeurIPS', 'ICLR', 'AAAI', 'IJCAI', 'CVPR', 'ICCV', 'ECCV',
                'SIGIR', 'WWW', 'SIGKDD', 'SIGMOD', 'VLDB', 'ICDE', 'OSDI', 'SOSP',
                'PLDI', 'POPL', 'OOPSLA', 'ISCA', 'MICRO', 'ASPLOS', 'CHI', 'UIST',
                'ICSE', 'FSE', 'ASE', 'ISSTA', 'CCS', 'NDSS', 'USENIX Security',
                'S&P', 'CRYPTO', 'EUROCRYPT', 'ASIACRYPT', 'TCC', 'STOC', 'FOCS'
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
                'Physical Review D', 'Physical Review E', 'Reviews of Modern Physics'
            ],
            'chemistry': [
                'Journal of the American Chemical Society', 'Angewandte Chemie',
                'Chemical Reviews', 'Nature Chemistry', 'Chemical Science',
                'Accounts of Chemical Research', 'Chemical Communications'
            ]
        }
    
    def search_papers(self, query: str, num_results: int = 50) -> List[Paper]:
        """Tìm kiếm papers từ Google Scholar API"""
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
                    break
                
                for result in data["organic_results"]:
                    paper = self._parse_paper(result)
                    if paper:
                        papers.append(paper)
                
                if len(data["organic_results"]) < results_per_page:
                    break
                    
                start += results_per_page
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Lỗi khi gọi API: {e}")
                break
        
        return papers[:num_results]
    
    def _parse_paper(self, result: Dict[str, Any]) -> Paper:
        """Parse result từ API thành Paper object"""
        try:
            title = result.get("title", "")
            pub_info = result.get("publication_info", {})
            summary = pub_info.get("summary", "")
            
            # Extract year
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
                title=title, authors=authors, year=year, venue=venue,
                citations=citations, snippet=snippet, link=link
            )
            
        except Exception as e:
            print(f"Lỗi khi parse paper: {e}")
            return None
    
    def calculate_relevance_score(self, paper: Paper, query: str) -> float:
        """Tính điểm độ liên quan"""
        query_terms = query.lower().split()
        text = (paper.title + " " + paper.snippet).lower()
        
        matches = sum(1 for term in query_terms if term in text)
        base_score = matches / len(query_terms)
        
        if query.lower() in text:
            base_score += 0.2
        
        citation_bonus = min(0.3, paper.citations / 1000) if paper.citations > 0 else 0
        return min(1.0, base_score + citation_bonus)
    
    def calculate_venue_score(self, paper: Paper, field: str = 'computer_science') -> float:
        """Tính điểm venue"""
        if field not in self.prestigious_venues:
            field = 'computer_science'
        
        prestigious_list = self.prestigious_venues[field]
        venue_lower = paper.venue.lower()
        
        for venue in prestigious_list:
            if venue.lower() in venue_lower:
                return 1.0
        
        for venue in prestigious_list:
            venue_words = venue.lower().split()
            if any(word in venue_lower for word in venue_words if len(word) > 3):
                return 0.7
        
        if paper.citations > 100:
            return 0.5
        elif paper.citations > 50:
            return 0.3
        
        return 0.1
    
    def calculate_recency_score(self, paper: Paper) -> float:
        """Tính điểm độ gần đây"""
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
        """Rerank papers theo các tiêu chí"""
        if weights is None:
            weights = {'relevance': 0.4, 'venue': 0.35, 'recency': 0.25}
        
        for paper in papers:
            paper.relevance_score = self.calculate_relevance_score(paper, query)
            paper.venue_score = self.calculate_venue_score(paper, field)
            paper.recency_score = self.calculate_recency_score(paper)
            
            paper.total_score = (
                paper.relevance_score * weights['relevance'] +
                paper.venue_score * weights['venue'] +
                paper.recency_score * weights['recency']
            )
        
        return sorted(papers, key=lambda p: p.total_score, reverse=True)
    
    def find_pdf_url(self, paper: Paper) -> str:
        """Tìm PDF URL từ paper link"""
        if not paper.link:
            return ""
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            if paper.link.lower().endswith('.pdf'):
                return paper.link
            
            # Handle arXiv
            if 'arxiv.org' in paper.link:
                if '/abs/' in paper.link:
                    return paper.link.replace('/abs/', '/pdf/') + '.pdf'
                elif '/pdf/' in paper.link:
                    return paper.link
            
            response = requests.get(paper.link, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Tìm PDF links
            pdf_patterns = [
                'a[href$=".pdf"]', 'a[href*=".pdf"]',
                'a[href*="arxiv.org/pdf/"]', 'a:contains("PDF")',
                'a:contains("Download PDF")', 'a:contains("Full Text PDF")'
            ]
            
            for pattern in pdf_patterns:
                elements = soup.select(pattern)
                for element in elements:
                    href = element.get('href', '')
                    if href:
                        if href.startswith('/'):
                            base_url = urllib.parse.urljoin(paper.link, '/')
                            href = urllib.parse.urljoin(base_url, href)
                        elif not href.startswith('http'):
                            href = urllib.parse.urljoin(paper.link, href)
                        
                        if self._is_valid_pdf_url(href):
                            return href
            
            # Check meta tags
            meta_pdf = soup.find('meta', {'name': 'citation_pdf_url'})
            if meta_pdf and meta_pdf.get('content'):
                return meta_pdf['content']
            
            return ""
            
        except Exception as e:
            print(f"Lỗi khi tìm PDF: {e}")
            return ""
    
    def _is_valid_pdf_url(self, url: str) -> bool:
        """Kiểm tra URL có phải PDF hợp lệ"""
        try:
            if not url.startswith('http'):
                return False
            return url.lower().endswith('.pdf') or 'pdf' in url.lower()
        except:
            return False
    
    def download_pdf(self, paper: Paper, download_dir: str = "papers") -> bool:
        """Tải PDF của paper"""
        if not paper.pdf_url:
            paper.pdf_url = self.find_pdf_url(paper)
        
        if not paper.pdf_url:
            return False
        
        try:
            Path(download_dir).mkdir(exist_ok=True)
            
            safe_title = re.sub(r'[<>:"/\\|?*]', '_', paper.title)[:100]
            filename = f"{safe_title}_{paper.year}.pdf"
            filepath = Path(download_dir) / filename
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
            response = requests.get(paper.pdf_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not paper.pdf_url.lower().endswith('.pdf'):
                return False
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            paper.pdf_path = str(filepath)
            paper.pdf_content = self.extract_pdf_text(str(filepath))
            
            return True
            
        except Exception as e:
            print(f"Lỗi khi tải PDF: {e}")
            return False
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Trích xuất text từ PDF"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:50000] if len(text) > 50000 else text
            
        except Exception as e:
            print(f"Lỗi khi đọc PDF: {e}")
            return ""
    
    def download_top_papers(self, papers: List[Paper], top_k: int = 5, 
                          download_dir: str = "papers") -> List[Paper]:
        """Tải PDF của top K papers"""
        top_papers = papers[:top_k]
        successful_downloads = []
        
        print(f"\n🔽 ĐANG TẢI TOP {top_k} PAPERS...")
        
        for i, paper in enumerate(top_papers, 1):
            try:
                print(f"[{i}/{top_k}] {paper.title[:60]}...")
                print(f"    Link: {paper.link}")
                
                paper.pdf_url = self.find_pdf_url(paper)
                print(f"    PDF URL: {paper.pdf_url}")
                
                if paper.pdf_url:
                    success = self.download_pdf(paper, download_dir)
                    if success:
                        successful_downloads.append(paper)
                        print(f"✅ Tải thành công: {paper.pdf_path}")
                    else:
                        print("❌ Không thể tải PDF")
                else:
                    print("❌ Không tìm thấy PDF URL")
                    
            except Exception as e:
                print(f"❌ Lỗi khi xử lý paper {i}: {e}")
            
            time.sleep(2)  # Rate limiting
        
        print(f"\n✅ Đã tải {len(successful_downloads)}/{top_k} papers")
        return successful_downloads


# ================================
# CREWAI TOOLS
# ================================

class PaperSearchInput(BaseModel):
    """Input schema cho paper search"""
    query: str = Field(..., description="Research query")
    field: str = Field(default="computer_science", description="Research field")
    num_results: int = Field(default=20, description="Number of papers")
    weights: Optional[Dict[str, float]] = Field(default=None, description="Ranking weights")


class PaperDownloadInput(BaseModel):
    """Input schema cho paper download"""
    papers_json: str = Field(..., description="JSON string of papers")
    top_k: int = Field(default=5, description="Number of papers to download")
    download_dir: str = Field(default="papers", description="Download directory")


class PaperSearchTool(BaseTool):
    """CrewAI Tool để tìm kiếm papers"""
    name: str = "paper_search"
    description: str = (
        "Tìm kiếm và rank papers từ Google Scholar. "
        "Input: query, field, num_results, weights. "
        "Output: JSON danh sách papers với scores."
    )
    args_schema: type[BaseModel] = PaperSearchInput
    
    def _run(self, query: str, field: str = "computer_science", 
             num_results: int = 20, weights: Optional[Dict[str, float]] = None) -> str:
        """Execute paper search"""
        try:
            # Create crawler instance
            crawler = ScholarCrawler()
            
            # Default weights cho từng field
            if weights is None:
                field_weights = {
                    'computer_science': {'relevance': 0.4, 'venue': 0.35, 'recency': 0.25},
                    'biology': {'relevance': 0.45, 'venue': 0.4, 'recency': 0.15},
                    'physics': {'relevance': 0.4, 'venue': 0.4, 'recency': 0.2},
                    'chemistry': {'relevance': 0.4, 'venue': 0.4, 'recency': 0.2}
                }
                weights = field_weights.get(field, field_weights['computer_science'])
            
            # Search papers
            papers = crawler.search_papers(query, num_results)
            
            # Rerank papers
            ranked_papers = crawler.rerank_papers(papers, query, field, weights)
            
            # Convert to dict
            papers_data = []
            for paper in ranked_papers:
                papers_data.append({
                    'title': paper.title,
                    'authors': paper.authors,
                    'venue': paper.venue,
                    'year': paper.year,
                    'citations': paper.citations,
                    'snippet': paper.snippet,
                    'link': paper.link,
                    'relevance_score': paper.relevance_score,
                    'venue_score': paper.venue_score,
                    'recency_score': paper.recency_score,
                    'total_score': paper.total_score
                })
            
            result = {
                'status': 'success',
                'query': query,
                'field': field,
                'total_papers': len(papers_data),
                'papers': papers_data
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({
                'status': 'error',
                'message': f"Lỗi: {str(e)}"
            }, ensure_ascii=False)


class PaperDownloadTool(BaseTool):
    """CrewAI Tool để tải PDF papers"""
    name: str = "paper_download"
    description: str = (
        "Tải PDF của top papers. "
        "Input: papers_json, top_k, download_dir. "
        "Output: JSON với papers đã tải PDF."
    )
    args_schema: type[BaseModel] = PaperDownloadInput
    
    def _run(self, papers_json: str, top_k: int = 5, download_dir: str = "papers") -> str:
        """Execute PDF download"""
        try:
            print(f"🔧 PaperDownloadTool: Starting download with top_k={top_k}")
            print(f"📄 Input papers_json length: {len(papers_json) if papers_json else 0}")
            
            # Create crawler instance
            crawler = ScholarCrawler()
            
            # Parse papers data
            try:
                papers_data = json.loads(papers_json)
                print(f"✅ Successfully parsed JSON with {len(papers_data.get('papers', []))} papers")
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error: {e}")
                return json.dumps({'status': 'error', 'message': f'JSON parsing error: {str(e)}'})
            
            if papers_data.get('status') != 'success':
                error_msg = f"Invalid papers data status: {papers_data.get('status')}"
                print(f"❌ {error_msg}")
                return json.dumps({'status': 'error', 'message': error_msg})
            
            # Convert to Paper objects
            papers = []
            for i, paper_dict in enumerate(papers_data['papers'][:top_k]):
                try:
                    paper = Paper(
                        title=paper_dict['title'],
                        authors=paper_dict['authors'],
                        venue=paper_dict['venue'],
                        year=paper_dict['year'],
                        citations=paper_dict['citations'],
                        snippet=paper_dict['snippet'],
                        link=paper_dict['link']
                    )
                    paper.relevance_score = paper_dict['relevance_score']
                    paper.venue_score = paper_dict['venue_score']
                    paper.recency_score = paper_dict['recency_score']
                    paper.total_score = paper_dict['total_score']
                    papers.append(paper)
                    print(f"✅ Converted paper {i+1}: {paper.title[:50]}...")
                except Exception as e:
                    print(f"❌ Error converting paper {i+1}: {e}")
                    continue
            
            print(f"📚 Total papers to download: {len(papers)}")
            
            # Download PDFs
            successful_downloads = crawler.download_top_papers(papers, top_k, download_dir)
            
            # Prepare result
            downloaded_papers = []
            for paper in successful_downloads:
                downloaded_papers.append({
                    'title': paper.title,
                    'authors': paper.authors,
                    'venue': paper.venue,
                    'year': paper.year,
                    'citations': paper.citations,
                    'pdf_url': paper.pdf_url,
                    'pdf_path': paper.pdf_path,
                    'pdf_content_length': len(paper.pdf_content),
                    'pdf_preview': paper.pdf_content[:500] if paper.pdf_content else "",
                    'total_score': paper.total_score
                })
            
            result = {
                'status': 'success',
                'requested': top_k,
                'downloaded': len(successful_downloads),
                'download_dir': download_dir,
                'papers': downloaded_papers
            }
            
            print(f"🎉 Download completed: {len(successful_downloads)}/{top_k} papers")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            error_msg = f"PaperDownloadTool error: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return json.dumps({
                'status': 'error',
                'message': error_msg
            }, ensure_ascii=False)


# ================================
# CREWAI AGENT & CREW
# ================================

def create_paper_crawler_agent() -> Agent:
    """Tạo CrewAI Agent cho paper crawling"""
    search_tool = PaperSearchTool()
    download_tool = PaperDownloadTool()
    
    agent = Agent(
        role='Paper Research Specialist',
        goal='Tìm kiếm, phân tích và tải xuống papers nghiên cứu liên quan đến query của user',
        backstory="""
        Bạn là chuyên gia nghiên cứu có kinh nghiệm sâu trong việc tìm kiếm và đánh giá papers khoa học.
        Bạn có khả năng:
        - Tìm kiếm papers từ Google Scholar với độ chính xác cao
        - Đánh giá và ranking papers theo độ liên quan, venue uy tín và thời gian
        - Tải xuống PDF và trích xuất nội dung từ papers
        - Phân tích và tóm tắt nội dung papers chi tiết
        
        Bạn luôn cung cấp kết quả có cấu trúc, chi tiết và hữu ích cho nghiên cứu.
        """,
        verbose=True,
        allow_delegation=False,
        tools=[search_tool, download_tool],
        max_iter=5,
        memory=True,
        llm=LLM_CLIENT
    )
    
    return agent


class PaperCrawlerCrew:
    """CrewAI Crew cho paper crawling workflow"""
    
    def __init__(self):
        self.agent = create_paper_crawler_agent()
    
    def search_papers(self, query: str, field: str = "computer_science", 
                     num_results: int = 20) -> str:
        """Tìm kiếm papers"""
        task = Task(
            description=f"""
            Tìm kiếm và ranking {num_results} papers liên quan đến query: "{query}" trong lĩnh vực {field}.
            
            Yêu cầu:
            1. Sử dụng paper_search tool
            2. Rank theo độ liên quan, venue uy tín, thời gian gần nhất
            3. Ưu tiên papers từ venues uy tín và gần đây
            4. Trả về JSON với danh sách papers và scores
            """,
            expected_output="JSON danh sách papers với metadata và scores",
            agent=self.agent
        )
        
        crew = Crew(agents=[self.agent], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        return str(result)
    
    def search_and_download(self, query: str, field: str = "computer_science", 
                          num_results: int = 20, top_k: int = 5) -> str:
        """Tìm kiếm và tải PDF papers"""
        search_task = Task(
            description=f"""
            Tìm kiếm {num_results} papers cho query: "{query}" trong {field}.
            Sử dụng paper_search tool và trả về JSON results.
            """,
            expected_output="JSON danh sách papers với scores",
            agent=self.agent
        )
        
        download_task = Task(
            description=f"""
            Tải PDF của top {top_k} papers từ kết quả search trước đó.
            
            HƯỚNG DẪN CHI TIẾT:
            1. Lấy kết quả JSON từ task search trước
            2. Sử dụng paper_download tool với parameters:
               - papers_json: JSON string từ task trước
               - top_k: {top_k}  
               - download_dir: "papers"
            3. Tool sẽ tự động tìm PDF URL và tải xuống
            4. Trả về JSON với thông tin papers đã tải
            
            LƯU Ý: Phải truyền chính xác JSON result từ search task vào papers_json parameter.
            """,
            expected_output="JSON với danh sách papers đã tải PDF thành công",
            agent=self.agent,
            context=[search_task]
        )
        
        crew = Crew(
            agents=[self.agent], 
            tasks=[search_task, download_task], 
            process=Process.sequential, 
            verbose=True
        )
        
        result = crew.kickoff()
        return str(result)


# ================================
# MANAGER INTEGRATION INTERFACE
# ================================

class PaperCrawlerInterface:
    """Interface cho Manager system"""
    
    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        self.crew = PaperCrawlerCrew()
        self.agent = self.crew.agent
        self.config = custom_config or {}
    
    def get_agent(self) -> Agent:
        """Trả về CrewAI Agent instance"""
        return self.agent
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Thông tin capabilities của agent"""
        return {
            "agent_type": "paper_crawler",
            "version": "1.0.0",
            "description": "Specialized agent for academic paper research",
            "capabilities": [
                "search_papers", "rank_papers", "download_pdfs", 
                "extract_content", "analyze_papers"
            ],
            "supported_fields": ["computer_science", "biology", "physics", "chemistry"],
            "tools": [
                {
                    "name": "paper_search",
                    "description": "Search and rank papers from Google Scholar"
                },
                {
                    "name": "paper_download", 
                    "description": "Download PDFs of top papers"
                }
            ]
        }
    
    def search_papers(self, query: str, **kwargs) -> Dict[str, Any]:
        """Tìm kiếm papers - interface cho Manager"""
        try:
            field = kwargs.get('field', 'computer_science')
            num_results = kwargs.get('num_results', 20)
            
            result_str = self.crew.search_papers(query, field, num_results)
            
            try:
                result_data = json.loads(result_str)
            except:
                result_data = {"status": "success", "raw_result": result_str}
            
            return {
                "agent_id": "paper_crawler",
                "task_type": "search_papers",
                "status": "completed",
                "input": {"query": query, "field": field, "num_results": num_results},
                "output": result_data
            }
            
        except Exception as e:
            return {
                "agent_id": "paper_crawler",
                "task_type": "search_papers", 
                "status": "error",
                "error": str(e)
            }
    
    def download_papers(self, papers_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Tải PDF papers"""
        try:
            top_k = kwargs.get('top_k', 5)
            download_dir = kwargs.get('download_dir', 'papers')
            
            papers_json = json.dumps(papers_data) if isinstance(papers_data, dict) else papers_data
            
            download_tool = PaperDownloadTool()
            result_str = download_tool._run(papers_json, top_k, download_dir)
            
            try:
                result_data = json.loads(result_str)
            except:
                result_data = {"status": "success", "raw_result": result_str}
            
            return {
                "agent_id": "paper_crawler",
                "task_type": "download_papers",
                "status": "completed",
                "output": result_data
            }
            
        except Exception as e:
            return {
                "agent_id": "paper_crawler",
                "task_type": "download_papers",
                "status": "error",
                "error": str(e)
            }
    
    def full_research_workflow(self, query: str, **kwargs) -> Dict[str, Any]:
        """Complete research workflow"""
        try:
            field = kwargs.get('field', 'computer_science')
            num_results = kwargs.get('num_results', 20)
            top_k = kwargs.get('top_k', 5)
            
            result_str = self.crew.search_and_download(query, field, num_results, top_k)
            
            try:
                result_data = json.loads(result_str)
            except:
                result_data = {"status": "success", "raw_result": result_str}
            
            return {
                "agent_id": "paper_crawler",
                "task_type": "full_research_workflow",
                "status": "completed",
                "input": {"query": query, "field": field, "num_results": num_results, "top_k": top_k},
                "output": result_data
            }
            
        except Exception as e:
            return {
                "agent_id": "paper_crawler",
                "task_type": "full_research_workflow",
                "status": "error",
                "error": str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Agent status"""
        return {
            "agent_id": "paper_crawler",
            "status": "ready",
            "version": "1.0.0",
            "last_updated": "2025-09-11"
        }


class PaperCrawlerFactory:
    """Factory để tạo Paper Crawler instances"""
    
    @staticmethod
    def create_agent(config: Optional[Dict[str, Any]] = None) -> PaperCrawlerInterface:
        """Tạo Paper Crawler Agent"""
        return PaperCrawlerInterface(config)
    
    @staticmethod
    def create_multiple_agents(count: int, configs: Optional[List[Dict[str, Any]]] = None) -> List[PaperCrawlerInterface]:
        """Tạo multiple agents"""
        agents = []
        for i in range(count):
            config = configs[i] if configs and i < len(configs) else None
            agents.append(PaperCrawlerInterface(config))
        return agents
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Default configuration"""
        return {
            "verbose": True,
            "memory": True,
            "max_iter": 5,
            "supported_fields": ["computer_science", "biology", "physics", "chemistry"]
        }
    
    @staticmethod
    def test_direct_search_and_download(query: str, top_k: int = 3) -> Dict[str, Any]:
        """Test search and download directly without CrewAI"""
        try:
            print(f"🧪 TESTING DIRECT SEARCH & DOWNLOAD")
            print(f"Query: {query}")
            print(f"Top K: {top_k}")
            
            # Direct search
            search_tool = PaperSearchTool()
            search_result = search_tool._run(query, "computer_science", 10)
            print(f"✅ Search completed")
            
            # Direct download  
            download_tool = PaperDownloadTool()
            download_result = download_tool._run(search_result, top_k, "papers")
            print(f"✅ Download completed")
            
            return {
                "status": "success",
                "search_result": json.loads(search_result),
                "download_result": json.loads(download_result)
            }
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e)
            }


# ================================
# EXAMPLE USAGE
# ================================

if __name__ == "__main__":
    print("=== PAPER CRAWLER CREWAI AGENT ===")
    
    # Tạo agent cho Manager
    crawler = PaperCrawlerFactory.create_agent()
    
    # Lấy thông tin agent
    capabilities = crawler.get_capabilities()
    print(f"Agent Type: {capabilities['agent_type']}")
    print(f"Capabilities: {capabilities['capabilities']}")
    
    # Example usage với CrewAI Agent
    query = "RAG technique"
    print(f"\n🔍 CrewAI Agent workflow with query: {query}")
    
 
    # Option 2: Full workflow (search + download)
    print(f"\n📚 Option 2: Full research workflow (search + download)...")
    full_result = crawler.full_research_workflow(
        query=query,
        field="computer_science",
        num_results=10,
        top_k=5
    )
