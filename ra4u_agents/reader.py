from crewai import LLM, Agent, Task, Crew, Process
import PyPDF2
import os
import re
from dotenv import load_dotenv

load_dotenv()

def read_from_disk(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)

        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text[:50000]

def get_agent(model="gemini/gemini-2.5-flash-preview-04-17"):
    client = LLM(model=model, api_key=os.getenv("GEMINI_API_KEY"))

    return Agent(
        role="Research Assistant",
        goal="Extract key information from a research paper, specifically related work and limitations.",
        backstory="You are an expert AI research assistant. Given the full text of a research paper, your goal is to identify and summarize relevant sections as requested by the user.",
        verbose=True,
        allow_delegation=False,
        llm=client,
    )

def run_reader(reader, doc_content):
    reading_task = Task(
        description=f"""
        Analyze the following research paper content and follow the instructions to extract information:
        ---
        DOCUMENT CONTENT:
        {doc_content}
        ---
        INSTRUCTIONS:
        {READER_PROMPT}
        """,
        agent=reader,
        expected_output=(
            "A document with two main sections: 'Related Works' and 'Limitations'. "
            "Each section must contain a clear, concise, and objective summary of the identified information. "
            "Ensure the output is well-structured and easy to read."
        ),
    )
    
    crew = Crew(
        agents=[reader],
        tasks=[reading_task],
        verbose=True,
        process=Process.sequential,
    )
    
    result = crew.kickoff()
    return result

def get_reader_result(folder_path):
    results = []
    reader_agent = get_agent()
    
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return []

    for document_name in os.listdir(folder_path):
        if document_name.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, document_name)
            pdf_content = read_from_disk(pdf_path)
            
            if pdf_content:
                print(f"Processing file: {document_name}")
                result = run_reader(reader=reader_agent, doc_content=pdf_content)
                results.append(result)
            else:
                print(f"Skipping empty file: {document_name}")

    return results

READER_PROMPT = """
1. Extract the "Related Work" section (or its equivalent, e.g., "Background," "Literature Review").
- Identify and summarize each referenced paper or approach mentioned.
- For each work, clearly state the main contribution or method.

2. Identify the limitations of the referenced works.
- Explicitly extract limitations if the paper states them.
- If the limitations aren't clearly stated, propose them based on the description (e.g., restricted dataset, lack of generalization, computational cost, domain-specific assumptions).
- Present them in a clear and concise way.

Be precise, concise, and objective. Do not include unrelated sections of the paper.
"""