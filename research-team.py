import os
import time
import requests
from datetime import datetime
from typing import Type
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from crewai_tools import TavilySearchTool

load_dotenv()

# --- STARTUP CHECK ---
# This runs immediately when the script launches, before any agents fire.
# It tells you whether the API key loaded correctly from your .env file.
_ss_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
if _ss_key:
    print(f"[STARTUP] Semantic Scholar API key loaded: ...{_ss_key[-6:]}")
else:
    print("[STARTUP] WARNING: No SEMANTIC_SCHOLAR_API_KEY found in .env — running unauthenticated.")

###############################################################
### THE CUSTOM SEMANTIC SCHOLAR TOOL
###
### This replaces TavilySearchTool. Instead of searching the
### general web, this tool queries the Semantic Scholar
### academic database directly.
###
### HOW A CUSTOM TOOL WORKS:
### CrewAI needs tools to follow a specific pattern so it
### knows what inputs the tool accepts and how to run it.
### There are three required pieces:
###
###   1. An Input Schema (SemanticScholarInput) — a class
###      that defines what arguments the tool accepts.
###      Think of it as the tool's intake form.
###
###   2. A Tool Class (SemanticScholarTool) — the tool itself,
###      which inherits from BaseTool. It has a name and
###      description (which the agent reads to decide when to
###      use the tool) and a _run() method that contains the
###      actual logic.
###
###   3. The _run() method — this is where the real work
###      happens. It calls the Semantic Scholar API, parses
###      the response, and returns a readable string that
###      the agent can reason about.
###############################################################

class SemanticScholarInput(BaseModel):
    """Defines the inputs this tool accepts."""
    query: str = Field(
        description="The search query. Can be a topic, a paper title, or an author name."
    )
    year_range: str = Field(
        default="",
        description="Optional year range to filter results, formatted as 'YYYY-YYYY'. "
                    "Example: '2015-2024' for recent work, '1985-2015' for foundational work. "
                    "Leave empty to search all years."
    )
    limit: int = Field(
        default=10,
        description="Number of results to return. Maximum is 20."
    )


class SemanticScholarTool(BaseTool):
    """The tool itself."""

    # The name and description are what the agent reads to
    # decide when and how to use this tool.
    name: str = "Semantic Scholar Academic Search"
    description: str = (
        "Searches the Semantic Scholar database of 200+ million peer-reviewed "
        "academic papers. Returns real papers with verified titles, authors, "
        "publication years, journals, citation counts, and abstracts. "
        "Use this for ALL source discovery and citation verification. "
        "Supports optional year range filtering."
    )
    args_schema: Type[BaseModel] = SemanticScholarInput

    def _run(self, query: str, year_range: str = "", limit: int = 10) -> str:
        """
        This method runs when an agent calls the tool.
        It contacts the Semantic Scholar API and formats
        the results into readable text.
        """

        # --- DIAGNOSTIC TRACE ---
        print(f"\n[SS TOOL] Query: '{query}' | Year range: '{year_range}' | Limit: {limit}")

        # --- BUILD THE API REQUEST ---
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

        # These are the data fields we want back for each paper.
        # Semantic Scholar won't return everything by default —
        # you have to ask for what you need.
        fields = "title,authors,year,publicationVenue,citationCount,abstract,externalIds"

        params = {
            "query": query,
            "fields": fields,
            "limit": min(limit, 20)  # API maximum is 100, but 20 is plenty
        }

        # Add year filter only if one was provided
        if year_range and "-" in year_range:
            params["year"] = year_range

        # If an API key exists in the .env file, use it.
        # If not, the request still works — just more slowly.
        headers = {}
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key

        # --- MAKE THE REQUEST (with retry on 429) ---
        # If Semantic Scholar rate-limits us, we wait and try again
        # rather than giving up immediately. Each retry waits a bit longer
        # than the last — this pattern is called exponential backoff.
        max_retries = 3
        for attempt in range(max_retries):
            try:
                time.sleep(1.1)
                response = requests.get(base_url, params=params, headers=headers, timeout=15)

                if response.status_code == 429:
                    wait_time = 5 * (attempt + 1)  # 5s, then 10s, then 15s
                    print(f"[SS TOOL] Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue  # go back to the top of the loop and try again

                response.raise_for_status()
                print(f"[SS TOOL] Response status: {response.status_code}")
                break  # success — exit the retry loop

            except requests.exceptions.Timeout:
                return "ERROR: The Semantic Scholar API did not respond in time. Try again."
            except requests.exceptions.HTTPError as e:
                return f"ERROR: Semantic Scholar returned an error: {e}"
            except requests.exceptions.RequestException as e:
                return f"ERROR: Could not reach Semantic Scholar: {e}"
        else:
            # This runs only if the loop exhausted all retries without breaking
            return "ERROR: Semantic Scholar rate limit persists after 3 retries. Try again later."

        # --- PARSE AND FORMAT THE RESULTS ---
        data = response.json()
        papers = data.get("data", [])
        print(f"[SS TOOL] Papers returned: {len(papers)}")

        if not papers:
            return f"No results found in Semantic Scholar for query: '{query}'"

        # Build a readable string the agent can work with.
        # Each paper becomes a numbered block of text.
        results = []
        for i, paper in enumerate(papers, 1):

            # Authors come back as a list of objects.
            # This line pulls out just the name from each one.
            authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])])

            # The publication venue (journal or conference) can
            # live in different fields depending on the paper type.
            venue = ""
            pub_venue = paper.get("publicationVenue")
            if pub_venue:
                venue = pub_venue.get("name", "")

            title = paper.get("title", "No title available")
            year = paper.get("year", "Year unknown")
            citations = paper.get("citationCount", 0)
            abstract = paper.get("abstract") or "No abstract available."

            # Trim very long abstracts so the agent's context
            # doesn't get overwhelmed.
            if len(abstract) > 400:
                abstract = abstract[:400] + "..."

            block = (
                f"{i}. TITLE: {title}\n"
                f"   AUTHORS: {authors}\n"
                f"   YEAR: {year}\n"
                f"   VENUE: {venue}\n"
                f"   CITATIONS: {citations}\n"
                f"   ABSTRACT: {abstract}\n"
            )
            results.append(block)

        header = f"Semantic Scholar results for '{query}':\n\n"
        return header + "\n".join(results)


# Create the tool instance that agents will actually use.
semantic_scholar_tool = SemanticScholarTool()

# Tavily searches the open web, catching sources that live outside
# Semantic Scholar's database: book reviews, edited volume chapters,
# recent conference proceedings, and working papers.
tavily_tool = TavilySearchTool()

###############################################################
### DEFINE THE BRAINS
###############################################################

# 1. The Factual Brain (Cold) - For finding, citing, and verifying
factual_llm = LLM(
    model="openrouter/google/gemini-2.0-flash-001",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.0  # Zero temperature — no creativity, no guessing
)

# 2. The Creative Brain (Warm) - For synthesizing and critiquing
creative_llm = LLM(
    model="openrouter/google/gemini-2.0-flash-001",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7  # Higher temperature to encourage novel connections
)

###############################################################
### DEFINE THE AGENTS
###############################################################

# 1. THE JUNIOR RESEARCHER: Focused on recent scholarship (last 10 years)
junior_researcher = Agent(
    role='Contemporary {field} Scout',
    goal="""Identify 10 of the most significant peer-reviewed scholarly works
    on {topic} published in the last 10 years within {field}. Prioritize
    citation impact, journal prestige, and methodological innovation.
    Flag any works that appear frequently cited by others.
    Format all citations in {citation_format} style.""",
    backstory="""You are a sharp, methodical research librarian with a doctorate
    in {field}, two years out of your program. You are deeply plugged into
    current academic conversations and know which journals set the agenda.
    You are skeptical of sources that are not peer-reviewed, and you
    double-check publication dates obsessively. You would rather return 8
    verified, high-quality sources than 10 uncertain ones. You always note
    when a recent source is already being widely cited — that signals it
    matters. Under NO circumstances will you generate, infer, or invent
    citations. Every source you list must come directly from a Semantic
    Scholar search result.""",
    tools=[semantic_scholar_tool],
    llm=factual_llm,
    verbose=True
)

# 2. THE SENIOR RESEARCHER: Focused on foundational and canonical texts
senior_researcher = Agent(
    role='Foundational {field} Bibliographer',
    goal="""Identify foundational scholarship on {topic} by mining citation
    counts and search results from the Semantic Scholar database.
    Format all citations in {citation_format} style.""",
    backstory="""You are a veteran academic who understands that the heart of
    scholarship is found in the bibliography. You use Semantic Scholar's
    citation counts as hard evidence of a text's influence — a paper cited
    500 times is not an opinion, it is a fact of the field. You prioritize
    works that appear repeatedly across different search queries, because
    recurrence signals genuine foundational status. You are looking for the
    seminal works that defined the field. Under NO circumstances will you
    generate, infer, or invent citations. Every source you list must come
    directly from a Semantic Scholar search result.""",
    tools=[semantic_scholar_tool],
    llm=factual_llm,
    verbose=True
)

# 3. THE WEB SCOUT: Finds sources outside the Semantic Scholar database
web_scout = Agent(
    role='Web-Based {field} Scout',
    goal="""Use open web search to find scholarly sources on {topic} within
    {field} that may not appear in academic databases — including book
    chapters, edited volume entries, recent conference papers, and working
    papers. Your job is to supplement, not duplicate, what the database
    researchers have already found. Format all citations in {citation_format}
    style.""",
    backstory="""You are a resourceful research assistant who knows that not
    all important scholarship lives inside a database. You are skilled at
    finding book chapters buried in edited volumes, conference papers that
    predate their journal publication, working papers from university
    repositories, and recent scholarship that hasn't yet accumulated enough
    citations to surface in Semantic Scholar. You are honest about the limits
    of what you find — if you cannot confirm a publication venue or author
    for a source, you flag it rather than guessing. You focus on sources
    that add something genuinely different to what the database researchers
    have already provided. Under NO circumstances will you generate, infer,
    or invent citations.""",
    tools=[tavily_tool],
    llm=factual_llm,
    verbose=True
)

# 4. THE FACT-CHECKER: Verifies every citation before it reaches the Professor
fact_checker = Agent(
    role='{field} Citation Verifier',
    goal="""Verify every citation produced by the Junior Researcher and the
    Senior Researcher by searching for each one in Semantic Scholar.
    Confirm that the title, authors, and publication venue all match.
    Remove any citation that cannot be confirmed.
    Produce a clean, verified list for the Professor to use.""",
    backstory="""You are a professional fact-checker with a background in
    academic librarianship. You treat every citation as guilty until proven
    innocent. For each citation, you search Semantic Scholar for the exact
    title. If the search returns a matching record with the correct author
    and venue, the citation passes. If no match can be found, or if key
    details conflict, the citation is cut — no exceptions, no patches.
    You keep a transparent record: your output lists every source that
    passed, and separately lists every source that was cut and why.
    A shorter list of real sources is always better than a longer list
    of uncertain ones.""",
    tools=[semantic_scholar_tool],
    llm=factual_llm,
    verbose=True
)

# 5. THE PROFESSOR: Focused on synthesis and intellectual narrative
professor = Agent(
    role='Distinguished {field} Professor',
    goal="""Write a rigorous, 4-paragraph literature review using ONLY the
    verified sources provided by the Fact-Checker. The review must show the
    intellectual development of {topic} over time — not merely list what
    sources exist, but explain how ideas evolved, competed, and built upon
    one another.""",
    backstory="""You are a full professor who has published three books and
    chaired a graduate program in {field}. You have a particular talent for
    identifying the conversation underneath the citations — the shared
    assumptions, the fault lines, the debates that never quite get resolved.
    You believe a literature review is not a bibliography with commentary;
    it is an argument about how a field thinks. You write in full, authoritative
    prose. You never use bullet points. You are direct about disagreement between
    scholars when it exists. You do not pad your writing with vague affirmations
    like 'this important work explores...' — instead you say what a text claims
    and what it does. Your fourth paragraph is always the sharpest: it names
    what the field consistently fails to examine.""",
    llm=creative_llm,
    verbose=True
)

# 6. THE EDITOR: Focused on identifying gaps and promising research angles
editor = Agent(
    role='Critical {field} Reviewer',
    goal="""Draft a frank, specific 'So What?' memo that identifies the most
    significant and actionable gaps in the {field} scholarship on {topic}.
    This memo should help a researcher decide whether and where to intervene
    in the conversation.""",
    backstory="""You are the editor-in-chief of the premier journal in {field},
    and you have rejected more manuscripts than most scholars have written.
    You are deeply familiar with what the field thinks it has covered versus
    what it has actually examined. When you read a literature review, you ask
    five questions:
    (1) Whose voices are missing — by gender, race, geography, or institution?
    (2) What methodologies have not been tried on this topic?
    (3) What time periods or populations have been underexamined?
    (4) What obvious comparison or connection has no one made?
    (5) What assumption does everyone share that no one has questioned?
    You write the memo as if you are advising a promising junior scholar who
    needs blunt, actionable guidance — not flattery.""",
    llm=creative_llm,
    verbose=True
)

# 7. THE TYPESETTER: Focused strictly on formatting and collation
typesetter = Agent(
    role='Managing Editor & Typesetter',
    goal="""Collate ALL verified research, the literature review, and the
    So What? memo into a single, perfectly formatted Markdown document.
    Use the verified source list produced by the Fact-Checker as your
    definitive citation list — do not add to it or subtract from it.""",
    backstory="""You are a meticulous copy editor and archivist for a
    top-tier academic journal. You have zero tolerance for truncation,
    summarizing, or approximation. You work exclusively from the verified
    source list — you do not re-introduce sources that the Fact-Checker
    removed, and you do not invent new ones. You count every citation before
    you begin. You preserve the exact formatting provided by the researchers.
    If something seems missing or corrupted, you say so explicitly in your
    output rather than silently filling in the gap. You are the final
    gatekeeper and your reputation depends on accuracy, not speed.""",
    llm=factual_llm,
    verbose=True
)

###############################################################
### DEFINE THE TASKS
###############################################################

task_recent = Task(
    description="""Use the Semantic Scholar search tool to find 10 of the most
    significant scholarly articles regarding {topic} published in the last
    10 years. Set the year_range parameter to cover the last 10 years.
    These articles must be situated within the field of {field}.
    Prioritize results with high citation counts, as these signal importance.
    Provide full citations in {citation_format} style and a brief summary
    of how each reflects current academic trends.""",
    expected_output="""A numbered list of exactly 10 contemporary citations
    in {citation_format} style, each followed by a brief summary. Every
    citation must come from a Semantic Scholar search result.""",
    agent=junior_researcher
)

task_foundational = Task(
    description="""Use the Semantic Scholar search tool to find foundational
    scholarship on {topic} within the field of {field}.

    Run at least two searches with different search terms to cast a wide net.
    Sort your results by citation count — the most-cited works are your
    strongest candidates for foundational status.

    Provide 10 foundational sources total (a mix of articles and, where
    Semantic Scholar surfaces them, books or book chapters). Format all
    citations in {citation_format} style and include a summary of each
    source's core argument and its influence on the field.""",
    expected_output="""A numbered list of 10 foundational citations in
    {citation_format} style, ranked by citation count, each with a summary
    of its argument and historical impact. Every citation must come from
    a Semantic Scholar search result.""",
    agent=senior_researcher
)

task_web_scout = Task(
    description="""Use web search to find scholarly sources on {topic} within
    {field} that supplement what the database researchers have already found.

    Focus specifically on source types that databases often miss:
    - Chapters in edited academic volumes
    - Recent conference papers and proceedings
    - Working papers and preprints from university repositories
    - Book-length studies that may not be fully indexed

    Run at least two searches using different search terms. For each source
    you find, record the full title, author(s), publication venue or book
    title, and year. If you cannot confirm any of these details, flag the
    source as unverified rather than guessing.

    Do not duplicate sources already found by the Junior or Senior Researcher.
    Provide citations in {citation_format} style.""",
    expected_output="""A numbered list of up to 10 supplementary citations
    in {citation_format} style from web sources, each with a brief note on
    why it adds something the database results may have missed. Any sources
    whose details could not be fully confirmed should be clearly flagged.""",
    agent=web_scout
)

task_verify = Task(
    description="""You will find THREE lists of citations in the conversation
    above: one from the Junior Researcher (recent database scholarship), one
    from the Senior Researcher (foundational database scholarship), and one
    from the Web Scout (web-sourced supplementary scholarship).

    For each citation across all three lists, use the Semantic Scholar search
    tool to search for the exact title. Confirm that the author and publication
    venue in the search result match what the researcher reported. For sources
    the Web Scout flagged as unverified, apply extra scrutiny — if Semantic
    Scholar cannot confirm them either, remove them.

    Before verifying, deduplicate all three lists. If the same source
    appears in more than one researcher's list, treat it as a single source.
    Verify it once. In the output, note all the researchers who found it
    in a single combined bracket, like [Junior Researcher, Web Scout].

    Your output must have two clearly labeled sections:

    VERIFIED SOURCES
    Each unique source that passed verification, listed once only, in
    {citation_format} style. After each citation, note in brackets which
    researcher(s) found it (e.g. [Junior Researcher] or [Senior Researcher,
    Web Scout]). If the same source appeared in all three lists, that is a
    strong signal of its importance — note it as [All Researchers].

    REMOVED SOURCES
    Each unique source that failed verification, listed once only, with a
    one-sentence explanation of what could not be confirmed.""",
    expected_output="""A two-section report: VERIFIED SOURCES (all unique
    confirmed citations in {citation_format} style, each listed once, with
    researcher attribution) and REMOVED SOURCES (all rejected citations
    with reasons).""",
    agent=fact_checker
)

task_analyze = Task(
    description="""Using ONLY the citations in the VERIFIED SOURCES section
    produced by the Fact-Checker, write a comprehensive 4-paragraph
    literature review.

    Paragraph 1: The foundational roots of the topic — what ideas came first,
    who established them, and what intellectual problems they were responding to.
    Draw primarily from the foundational sources.

    Paragraph 2: The current state of the field — what questions are being
    asked now, what methods are being used, and what debates are live.
    Draw primarily from the recent sources.

    Paragraph 3: The arc of development — how contemporary scholarship
    diverges from, revises, or builds upon the foundational work. Be specific
    about which texts are in dialogue with which.

    Paragraph 4: The research gap — something consistently missing or
    unexamined across both eras that a new article could meaningfully address.
    Be direct and specific. Do not hedge.""",
    expected_output="""A 4-paragraph literature review written in full academic
    prose, with no bullet points, citing only verified sources.""",
    agent=professor
)

task_critique = Task(
    description="""Using the verified source list from the Fact-Checker AND
    the literature review written by the Professor, draft a 'So What?' memo
    for a junior scholar considering entering this conversation.
    Your memo must address all five of the following dimensions of absence:
    (1) Whose voices are missing by gender, race, geography, or institution?
    (2) What methodologies have not been applied to this topic?
    (3) What time periods or populations have been underexamined?
    (4) What obvious comparison or connection has no one made?
    (5) What shared assumption has gone unquestioned?
    Be blunt and specific. Identify the single most promising angle for
    a new contribution.""",
    expected_output="""A structured 'So What?' memo organized around the five
    gap dimensions, with a clear final recommendation.""",
    agent=editor
)

output_filename = datetime.now().strftime("litreview_%y%m%d%H%M%S.md")

task_compile = Task(
    description="""All of the content you need is already present in the
    conversation above from your colleagues. Do NOT ask for input. Do NOT
    prompt the user for anything. Do NOT wait for additional information.
    Begin compiling immediately using what is already in front of you.

    Locate these four items in the conversation above and assemble them
    into a single Markdown document:

    1. The VERIFIED SOURCES section from the Fact-Checker's output.
    2. The REMOVED SOURCES section from the Fact-Checker's output.
    3. The complete 4-paragraph literature review from the Professor's output.
    4. The complete So What? memo from the Editor's output.

    Structure the document with EXACTLY these four sections:

    # 1. Sources
    Copy the VERIFIED SOURCES list exactly as the Fact-Checker wrote it.
    Do not add, remove, or reformat any citation.
    Alphabetize the list by the author's last name.

    # 2. Removed Sources
    Copy the REMOVED SOURCES list exactly as the Fact-Checker wrote it,
    including the reason each source was removed.

    # 3. Literature Review
    Copy the Professor's literature review exactly. Do not alter it.

    # 4. So What? Memo
    Copy the Editor's So What? memo exactly. Do not alter it.""",
    expected_output="""A perfectly formatted Markdown document with all four
    sections: verified sources, removed sources, the complete literature
    review, and the complete So What? memo.""",
    agent=typesetter,
    output_file=output_filename
)

###############################################################
### ASSEMBLE THE CREW
###############################################################

research_crew = Crew(
    agents=[junior_researcher, senior_researcher, web_scout, fact_checker, professor, editor, typesetter],
    tasks=[task_recent, task_foundational, task_web_scout, task_verify, task_analyze, task_critique, task_compile],
    process=Process.sequential
)

###############################################################
### LAUNCH
###############################################################

print("=" * 60)
print("         MULTI-AGENT RESEARCH ASSISTANT")
print("         Semantic Scholar + Web Search")
print("=" * 60)
field_input = input("\nEnter the general field of study\n(e.g., American Literature, SoTL, Academic Administration): ").strip()
topic_input = input("\nEnter your research topic or hunch\n(e.g., 'code-switching in HBCU writing classrooms'): ").strip()
citation_input = input("\nEnter your preferred citation format\n(e.g., MLA, APA, Chicago): ").strip()

print(f"\n### Launching the Research Team ###")
print(f"    Topic:  {topic_input}")
print(f"    Field:  {field_input}")
print(f"    Format: {citation_input}\n")

result = research_crew.kickoff(inputs={
    'field': field_input,
    'topic': topic_input,
    'citation_format': citation_input
})

print("\n" + "=" * 60)
print("Task Complete!")
print(f"Your report has been saved to '{output_filename}'")
print("=" * 60)
