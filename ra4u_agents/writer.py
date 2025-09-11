from typing import Optional, List, Union

from crewai import Agent, Task

from agents import get_llm_client


def create_writer_agent() -> Agent:

	client = get_llm_client()

	return Agent(
		role="Writer",
		goal=(
			"Transform the Reader's extracted findings into a coherent, well-structured"
			" research report in markdown. Follow the exact section order and provide"
			" clear, concise, and academically toned writing."
		),
		backstory=(
			"An academic technical writer skilled at synthesizing related works,"
			" limitations, and articulating research gaps with actionable future directions."
		),
		verbose=True,
		allow_delegation=False,
		llm=client,
	)


def _build_writer_instructions(reader_output: Union[str, List[str]], title: Optional[str]) -> str:

	title_text = title.strip() if title else "Auto-Generated Research Report"

	if isinstance(reader_output, list):
		num_items = len(reader_output)
		joined_reader = "\n\n".join(reader_output)
	else:
		num_items = None
		joined_reader = reader_output

	return (
		"You are given the output from the Reader agent (extracted related works and"
		" limitations). Using ONLY that content as factual basis, write a concise"
		" research report in markdown with the following sections and exact headings:\n\n"
		f"# {title_text}\n\n"
		"## Introduction\n"
		"- Briefly set context and problem background in 3-6 sentences.\n\n"
		"## Related works\n"
		f"- Synthesize insights across ALL {num_items if num_items else ''} works collectively (themes, methods, outcomes).".rstrip() + "\n"
		"- Compare/contrast approaches, highlight commonalities and divergences.\n"
		"- If sources/links are present in the Reader output, include inline citations.\n"
		"- Avoid per-paper mini-summaries; focus on cross-paper synthesis.\n\n"
		"## Limitation\n"
		f"- Aggregate limitations across the {num_items if num_items else ''} works (methodology, data, evaluation, generalization).".rstrip() + "\n"
		"- Deduplicate and group similar limitations; provide a concise, collective view.\n\n"
		"## Research Gap & Future Research\n"
		"- Derive gaps from the cross-paper synthesis and aggregated limitations.\n"
		"- Propose 3-6 concrete, testable future research directions.\n"
		"- Each direction should be one bullet, clearly scoped and actionable.\n\n"
		"Writing and formatting requirements:\n"
		"- Use markdown only with clear section structure.\n"
		"- Academic, precise, and concise style; prefer active voice and strong topic sentences.\n"
		"- Maintain coherence with smooth transitions; avoid redundancy and vague claims.\n"
		"- Use consistent terminology; define key terms briefly if needed.\n"
		"- Avoid fabricating data not present in the Reader output.\n\n"
		f"Reader output (sole factual context). It may contain {num_items if num_items else 'multiple'} items, each with related works and limitations:\n"
		"-----\n"
		f"{joined_reader}\n"
		"-----\n"
	)


def create_writer_task(reader_output: Union[str, List[str]], *, title: Optional[str] = None) -> Task:

	writer = create_writer_agent()

	description = _build_writer_instructions(reader_output=reader_output, title=title)

	return Task(
		description=description,
		agent=writer,
		expected_output=(
			"A markdown document with sections: Introduction, Related works, Limitation,"
			" Research Gap & Future Research."
		),
	)


def run_writer(reader_output: Union[str, List[str]], *, title: Optional[str] = None) -> Task:

	# Backwards-compatible alias that now returns a Task instead of executing a Crew
	return create_writer_task(reader_output, title=title)


def run_writer_from_list(reader_outputs: List[str], *, title: Optional[str] = None) -> Task:

	return create_writer_task(reader_outputs, title=title)


