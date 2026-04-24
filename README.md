# Multi-Agent Academic Research Assistant

A multi-agent AI research assistant built with CrewAI that generates academic literature reviews. Six specialized agents — researchers, a fact-checker, a synthesizer, a gap analyst, and a typesetter — query Semantic Scholar and the web to produce verified, cited, publication-ready reports.

---

## What It Does

You provide a **field of study**, a **research topic**, and a **citation format**. The system assembles a research team that works sequentially to produce a structured Markdown report containing:

1. A verified source list drawn from academic databases and web search
2. A list of any sources that failed verification and were removed
3. A four-paragraph literature review tracing the intellectual development of the topic
4. A "So What?" memo identifying gaps in the scholarship and recommending the single most promising angle for a new contribution

---

## The Agents

| Agent | Role | Tool |
|---|---|---|
| **Junior Researcher** | Finds recent peer-reviewed scholarship (last 10 years) | Semantic Scholar |
| **Senior Researcher** | Finds foundational and canonical scholarship | Semantic Scholar |
| **Web Scout** | Finds sources outside academic databases (book chapters, conference papers, working papers) | Tavily Web Search |
| **Fact-Checker** | Verifies every citation independently; removes anything that cannot be confirmed | Semantic Scholar |
| **Professor** | Synthesizes verified sources into a literature review | — |
| **Editor** | Identifies gaps in the scholarship; writes the "So What?" memo | — |
| **Typesetter** | Collates all outputs into a single formatted Markdown document | — |

The agents run sequentially. Each agent receives the full output of all prior agents before it begins.

---

## Requirements

- Python 3.10+
- A [CrewAI](https://www.crewai.com/) installation
- An [OpenRouter](https://openrouter.ai/) API key (used to access the LLM)
- A [Tavily](https://tavily.com/) API key (used for web search)
- A [Semantic Scholar](https://www.semanticscholar.org/product/api) API key (recommended; the tool will function without one but at a much stricter rate limit)

### Python dependencies

```
pipenv install crewai crewai-tools python-dotenv requests
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Create your `.env` file

Create a file named `.env` in the project root with the following contents:

```
OPENROUTER_API_KEY=your_openrouter_key_here
TAVILY_API_KEY=your_tavily_key_here
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key_here
```

No quotes around the values. No spaces around the equals signs. The Semantic Scholar key is optional but strongly recommended — without it, the tool will hit rate limits quickly.

### 3. Run the script

```bash
pipenv run python research-team.py
```

You will be prompted for three inputs:

```
Enter the general field of study
(e.g., American Literature, SoTL, Academic Administration):

Enter your research topic or hunch
(e.g., 'code-switching in HBCU writing classrooms'):

Enter your preferred citation format
(e.g., MLA, APA, Chicago):
```

---

## Output

The report is saved automatically to the directory where the script is run, with a timestamped filename:

```
litreview_YYMMDDHHMMSS.md
```

This means each run produces a new file and previous reports are never overwritten.

---

## Known Limitations

This tool is designed to produce a **starting point**, not a comprehensive literature review. Be aware of the following blind spots:

- **Book-length scholarship** is underrepresented. Semantic Scholar indexes papers; monographs and edited volumes surface inconsistently.
- **Humanities journals** are less well-covered than STEM fields in Semantic Scholar.
- **Non-English scholarship** is largely invisible to the current search tools.
- **Verification creates a closed loop.** The Fact-Checker verifies sources by searching Semantic Scholar — meaning sources that exist but aren't in Semantic Scholar's database may be cut even if they are real and important.

Cross-check the output against **MLA International Bibliography**, **Project MUSE**, or **Google Scholar** for more complete coverage, particularly in literary and cultural studies.

---

## Customization

- **LLM model:** The model is set in the `factual_llm` and `creative_llm` definitions near the top of the script. Any model available through OpenRouter can be substituted.
- **Temperature:** The factual agents run at `temperature=0.0` to minimize hallucination. The creative agents (Professor and Editor) run at `temperature=0.7` to encourage synthesis and novel connections.
- **Source counts:** The number of sources each researcher is asked to find can be adjusted in the task descriptions.
- **Rate limiting:** The Semantic Scholar tool includes a 1.1-second delay between requests to stay within API rate limits. This can be adjusted in the `_run` method of `SemanticScholarTool`.

---

## License

MIT License. Use freely; attribution appreciated.
