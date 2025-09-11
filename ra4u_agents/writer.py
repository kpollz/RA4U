from typing import Optional

from crewai import Agent, LLM
import os


def create_writer_agent(*, llm: Optional[LLM] = None, model: Optional[str] = None) -> Agent:

	client = llm if llm is not None else LLM(model=model or os.getenv("WRITER_LLM_MODEL", "gemini/gemini-2.0-flash"))

	return Agent(
		role="Writer",
		goal=(
			"Transform the Reader's extracted findings into a coherent, well-structured"
			" research report in markdown using ONLY the Reader output as the factual"
			" basis. Follow the exact section order and adhere to these requirements:"
			"\n\n"
			"Requirement: Write the report following the structure: Introduction, Related works,"
			" Limitation, Research Gap & Future Research." "\n\n"
			"Sections and exact headings:" "\n"
			"- # <Title provided by task>" "\n"
			"- ## Introduction: Briefly set context and background in 3-6 sentences." "\n"
			"- ## Related works: Synthesize insights across ALL works collectively (themes,"
			" methods, outcomes). Compare/contrast approaches, highlight commonalities and"
			" divergences. If sources/links are present, include inline citations. Avoid"
			" per-paper mini-summaries; focus on cross-paper synthesis." "\n"
			"- ## Limitation: Aggregate limitations across the works (methodology, data,"
			" evaluation, generalization). Deduplicate and group similar limitations;"
			" provide a concise, collective view." "\n"
			"- ## Research Gap & Future Research: Derive gaps from the synthesis and"
			" limitations. Propose 3-6 concrete, testable future research directions; each"
			" direction should be one bullet, clearly scoped and actionable." "\n\n"
			"Writing and formatting requirements:" "\n"
			"- Use markdown only with clear section structure." "\n"
			"- Academic, precise, and concise style; prefer active voice and strong topic"
			" sentences." "\n"
			"- Maintain coherence with smooth transitions; avoid redundancy and vague"
			" claims." "\n"
			"- Use consistent terminology; define key terms briefly if needed." "\n"
			"- Do not fabricate data not present in the Reader output."
		),
		backstory=(
			"An academic technical writer skilled at synthesizing related works and"
			" limitations into rigorous narratives. You compare and contrast methods,"
			" surface themes and divergences, deduplicate limitations, and translate these"
			" into clearly scoped research gaps and actionable future directions."
		),
		verbose=True,
		allow_delegation=False,
		llm=client,
	)
