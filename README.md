# RoleMatcherAI

RoleMatcherAI is a Streamlit app for reviewing public job-posting pages. It extracts
the job description and structured job-posting details when available, detects
common AI/software/full-stack engineer role titles, and highlights a built-in list
of technologies and skills.

The app analyzes public HTML pages only. It does not read browser history, cookies,
or other private browser data. Pages that require a login or client-side rendering
may not expose their full job details to this first version.

## Run locally

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Paste a public `http://` or `https://` job-posting URL into the app and select
**Analyze job posting**.

## Index the portfolio in ChromaDB

Portfolio indexing is kept separate from the Streamlit app in
[`chroma-db-setup.py`](./chroma-db-setup.py). It reads `Data/portfolio.txt`,
splits the Markdown sections into overlapping text chunks, and stores them in a
persistent local ChromaDB collection named `portfolio`. Re-running the indexer
updates the collection and removes obsolete chunks.

1. Keep your portfolio file at `Data/portfolio.txt`. The file is intentionally
   ignored by Git; do not commit personal portfolio data or credentials.
2. From the project root, install the ChromaDB dependency. If you also use the
   Streamlit app, its complete dependencies are in `requirements.txt`; the
   indexing-only dependency list is `requirements-chroma.txt`.

   ```powershell
   py -3.12 -m venv .venv-chroma
   .\.venv-chroma\Scripts\Activate.ps1
   python -m pip install -r requirements-chroma.txt
   ```

   If the Python launcher cannot find Python 3.12, install Python 3.12 first.
   In future terminals, activate this environment before running the indexer:

   ```powershell
   .\.venv-chroma\Scripts\Activate.ps1
   ```

3. Build or refresh the local vector database:

   ```powershell
   python chroma-db-setup.py
   ```

   ChromaDB writes its persistent files to `chroma_db/`, which is ignored by
   Git. The first run may download ChromaDB's default embedding model and needs
   internet access.

To run the ChromaDB server locally against that same database:

```powershell
chroma run --path .\chroma_db --host 127.0.0.1 --port 8000
```

For deployment, build the index in the deployment environment from a securely
provided portfolio file, or mount `chroma_db/` on persistent storage. Do not
commit the database or expose an unauthenticated ChromaDB server publicly.
Only `chroma-db-setup.py` and `requirements-chroma.txt` are needed for the
standalone indexing step. The Streamlit app also uses ChromaDB to compare jobs
with the indexed portfolio, so install its dependencies from `requirements.txt`.

## Match jobs and create a tailored resume

The app compares job skills with the indexed portfolio and shows matched skills
and relevant portfolio sections. Resume generation is offered only for a strong
match (at least three matching listed skills and at least 60% skill coverage).
To enable it, add your Groq API key to the root `.env` file:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

For deployment, configure `GROQ_API_KEY` as a platform environment variable
instead of committing the `.env` file.

In the app, optionally upload a text-based PDF resume along with the job URL.
After analysis, select **Create tailored resume** and download the generated
PDF. Scanned PDFs require OCR first. The uploaded resume is held in the app
session; when generating, the job description, relevant portfolio excerpts, and
uploaded resume text are sent to Groq. Review the generated resume for accuracy
before using it.
