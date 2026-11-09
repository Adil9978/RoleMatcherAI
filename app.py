import html
import io
import ipaddress
import json
import os
import re
import socket
from urllib.parse import urljoin, urlsplit

import chromadb
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq, GroqError
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

load_dotenv()


MAX_PAGE_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5
REQUEST_TIMEOUT = (5, 15)
USER_AGENT = "RoleMatcherAI/1.0 (job posting analyzer)"
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
CHROMA_COLLECTION = "portfolio"
GROQ_MODEL = "openai/gpt-oss-20b"
MAX_RESUME_BYTES = 10 * 1024 * 1024
MAX_RESUME_PAGES = 20
MAX_RESUME_CHARS = 40_000
STRONG_MATCH_THRESHOLD = 60

ROLE_PATTERNS = {
    "AI Engineer": r"\b(?:AI|artificial intelligence|machine learning|ML)\s+(?:engineer|developer)\b",
    "Software Engineer": r"\bsoftware engineer\b",
    "Full Stack Engineer": r"\bfull[\s-]?stack (?:engineer|developer)\b",
    "Senior Full Stack Engineer": r"\bsenior full[\s-]?stack (?:engineer|developer)\b|\bsr\.? full[\s-]?stack (?:engineer|developer)\b",
    "Senior Software Engineer": r"\bsenior software engineer\b|\bsr\.? software engineer\b",
}

SKILL_TERMS = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Ruby",
    "PHP", "SQL", "React", "Angular", "Vue", "Node.js", "Django", "Flask",
    "FastAPI", "Spring", "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes",
    "Git", "Linux", "REST", "GraphQL", "Machine Learning", "Deep Learning",
    "TensorFlow", "PyTorch", "LangChain", "LLM", "NLP", "CI/CD",
]


class PageFetchError(Exception):
    pass


def _validate_public_url(url):
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise PageFetchError("Enter a complete public http:// or https:// URL.")
    if parts.username or parts.password:
        raise PageFetchError("URLs containing embedded credentials are not supported.")
    if parts.port and parts.port not in {80, 443}:
        raise PageFetchError("Only standard HTTP and HTTPS ports are supported.")

    host = parts.hostname.rstrip(".")
    try:
        addresses = {ipaddress.ip_address(host)}
    except ValueError:
        try:
            addresses = {
                ipaddress.ip_address(result[4][0])
                for result in socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80))
            }
        except (OSError, ValueError) as error:
            raise PageFetchError("The URL host could not be resolved.") from error

    if not addresses or any(not address.is_global for address in addresses):
        raise PageFetchError("Only publicly reachable websites can be analyzed.")
    return url


def fetch_page(url):
    current_url = url
    with requests.Session() as session:
        for redirect_count in range(MAX_REDIRECTS + 1):
            _validate_public_url(current_url)
            try:
                response = session.get(
                    current_url,
                    headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
                    timeout=REQUEST_TIMEOUT,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.RequestException as error:
                raise PageFetchError(f"Could not retrieve the page: {error}") from error

            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                response.close()
                if not location or redirect_count == MAX_REDIRECTS:
                    raise PageFetchError("The page redirected too many times or returned an invalid redirect.")
                current_url = urljoin(current_url, location)
                continue

            if response.status_code >= 400:
                response.close()
                raise PageFetchError(f"The website returned HTTP {response.status_code}.")

            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                response.close()
                raise PageFetchError("That URL did not return an HTML page.")

            content = bytearray()
            try:
                for chunk in response.iter_content(chunk_size=65536):
                    content.extend(chunk)
                    if len(content) > MAX_PAGE_BYTES:
                        raise PageFetchError("The page is larger than the 5 MB analysis limit.")
            finally:
                response.close()
            return bytes(content), response.encoding or "utf-8"

    raise PageFetchError("Could not retrieve the page.")


def _walk_job_postings(value):
    if isinstance(value, dict):
        item_type = value.get("@type", [])
        if isinstance(item_type, str):
            item_type = [item_type]
        if any(str(kind).rsplit("/", 1)[-1] == "JobPosting" for kind in item_type):
            yield value
        for child in value.values():
            yield from _walk_job_postings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_job_postings(child)


def _plain_text(value):
    if isinstance(value, str):
        return BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
    if isinstance(value, dict):
        return _plain_text(value.get("name") or value.get("value") or "")
    if isinstance(value, list):
        return ", ".join(filter(None, (_plain_text(item) for item in value)))
    return ""


def _organization_name(value):
    if isinstance(value, dict):
        return value.get("name", "")
    return value if isinstance(value, str) else ""


def _job_location(value):
    locations = value if isinstance(value, list) else [value]
    formatted = []
    for location in locations:
        if not isinstance(location, dict):
            text = _plain_text(location)
        else:
            address = location.get("address", location)
            if isinstance(address, dict):
                text = ", ".join(
                    str(address[key])
                    for key in ("streetAddress", "addressLocality", "addressRegion", "postalCode", "addressCountry")
                    if address.get(key)
                )
            else:
                text = _plain_text(address)
        if text and text not in formatted:
            formatted.append(text)
    return ", ".join(formatted)


def extract_job_details(html, encoding="utf-8"):
    soup = BeautifulSoup(html.decode(encoding, errors="replace"), "html.parser")
    posting = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or script.get_text())
        except (json.JSONDecodeError, TypeError):
            continue
        posting = next(_walk_job_postings(data), {})
        if posting:
            break

    for element in soup(["script", "style", "noscript", "svg", "iframe"]):
        element.decompose()

    visible_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    headings = list(dict.fromkeys(
        re.sub(r"\s+", " ", heading.get_text(" ", strip=True)).strip()
        for heading in soup.find_all(["h1", "h2", "h3"])
        if heading.get_text(" ", strip=True).strip()
    ))

    description = _plain_text(posting.get("description", ""))
    title = _plain_text(posting.get("title", ""))
    if not title:
        title = soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else ""
    if not title and soup.title:
        title = soup.title.get_text(" ", strip=True)
    if not title and headings:
        title = headings[0]
    if not description:
        description = visible_text

    roles = [
        role for role, pattern in ROLE_PATTERNS.items()
        if re.search(pattern, title, re.IGNORECASE)
    ]
    searchable_text = f"{title} {description} {visible_text}"
    skills = [
        skill for skill in SKILL_TERMS
        if re.search(rf"(?<![\w+#]){re.escape(skill)}(?![\w+#])", searchable_text, re.IGNORECASE)
    ]

    return {
        "title": title or "Not found",
        "company": _organization_name(posting.get("hiringOrganization")) or "Not found",
        "location": _job_location(posting.get("jobLocation", "")) or "Not found",
        "employment_type": _plain_text(posting.get("employmentType")) or "Not found",
        "roles": roles,
        "skills": skills,
        "headings": headings,
        "description": description,
        "visible_text": visible_text,
        "structured_data_found": bool(posting),
    }


def analyze_portfolio_match(details):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection_names = {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }
    if CHROMA_COLLECTION not in collection_names:
        raise RuntimeError(
            "The portfolio ChromaDB collection was not found. Run "
            "`python chroma-db-setup.py` before analyzing a job."
        )
    collection = client.get_collection(CHROMA_COLLECTION)
    if collection.count() == 0:
        raise RuntimeError("The portfolio ChromaDB collection is empty. Run `python chroma-db-setup.py`.")

    documents = collection.get(include=["documents", "metadatas"])
    portfolio_text = "\n".join(documents["documents"])
    portfolio_skills = {
        skill for skill in SKILL_TERMS
        if re.search(rf"(?<![\w+#]){re.escape(skill)}(?![\w+#])", portfolio_text, re.IGNORECASE)
    }
    job_skills = set(details["skills"])
    matched_skills = sorted(job_skills & portfolio_skills, key=str.casefold)
    missing_skills = sorted(job_skills - portfolio_skills, key=str.casefold)
    skill_score = round(100 * len(matched_skills) / len(job_skills)) if job_skills else 0

    query = " ".join((details["title"], details["description"]))[:6000]
    relevant = collection.query(
        query_texts=[query],
        n_results=min(5, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    evidence = []
    seen = set()
    for doc, metadata in zip(relevant["documents"][0], relevant["metadatas"][0]):
        section = metadata.get("section", "Portfolio")
        key = (section, doc)
        if key not in seen:
            evidence.append({"section": section, "text": doc})
            seen.add(key)

    role_title = details["title"].casefold()
    role_relevant = any(
        term in role_title
        for term in ("engineer", "developer", "machine learning", "artificial intelligence")
    )
    strong_match = (
        len(matched_skills) >= 3
        and skill_score >= STRONG_MATCH_THRESHOLD
        and bool(evidence)
        and role_relevant
    )
    return {
        "job_skills": sorted(job_skills, key=str.casefold),
        "portfolio_skills": sorted(portfolio_skills, key=str.casefold),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_score": skill_score,
        "strong_match": strong_match,
        "evidence": evidence,
    }


def extract_resume_text(pdf_bytes):
    if len(pdf_bytes) > MAX_RESUME_BYTES:
        raise ValueError("The uploaded resume must be 10 MB or smaller.")

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if reader.is_encrypted:
            raise ValueError("Password-protected resumes are not supported.")
        if len(reader.pages) > MAX_RESUME_PAGES:
            raise ValueError(f"The uploaded resume must have {MAX_RESUME_PAGES} pages or fewer.")
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except (PdfReadError, OSError) as error:
        raise ValueError(f"Could not read the uploaded PDF: {error}") from error
    except ValueError:
        raise

    if not text:
        raise ValueError("No selectable text was found in the PDF. Scanned PDFs need OCR first.")
    if len(text) > MAX_RESUME_CHARS:
        text = text[:MAX_RESUME_CHARS]
    return text


def create_tailored_resume(details, match, resume_text=None):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it to your local .env file "
            "or set it as an environment variable, then restart the app."
        )

    portfolio_evidence = "\n\n".join(
        f"Portfolio section: {item['section']}\n{item['text']}"
        for item in match["evidence"]
    )
    source_resume = resume_text or "No existing resume was uploaded. Use only supported facts from the portfolio."
    prompt = f"""Create a polished, ATS-friendly resume tailored to the job below.

Accuracy rules:
- Use only facts explicitly supported by the provided existing resume or portfolio evidence.
- Do not invent employers, dates, degrees, achievements, metrics, skills, certifications, or contact details.
- When an existing resume is supplied, preserve its real contact details and employment history; do not replace them with portfolio details that conflict.
- Prioritize evidence that matches the role and job requirements. Do not claim missing job skills.
- Omit unknown information instead of adding placeholders.
- Return plain text only, with a concise professional summary followed by clearly titled sections and bullet points.

JOB TITLE: {details['title']}
COMPANY: {details['company']}
JOB DESCRIPTION:
{details['description'][:12000]}

PORTFOLIO SKILLS THAT MATCH THE JOB: {", ".join(match['matched_skills']) or "None identified"}
PORTFOLIO EVIDENCE:
{portfolio_evidence[:12000]}

EXISTING RESUME:
{source_resume[:MAX_RESUME_CHARS]}
"""
    try:
        response = Groq(api_key=api_key).chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You create truthful, professional resumes and never fabricate candidate information.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=2500,
        )
    except GroqError as error:
        raise RuntimeError(f"Groq request failed: {error}") from error
    resume = response.choices[0].message.content
    if not resume or not resume.strip():
        raise RuntimeError("Groq returned an empty resume. Please try again.")
    resume = resume.strip()
    if resume.startswith("```"):
        resume = re.sub(r"^```[^\r\n]*\r?\n|\r?\n```$", "", resume).strip()
    return resume


def resume_to_pdf(resume_text, title):
    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ResumeTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#15324b"),
        spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        name="ResumeHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#176b87"),
        spaceBefore=10,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="ResumeBody",
        parent=styles["BodyText"],
        leading=14,
        spaceAfter=5,
    ))

    story = []
    title_added = False
    for raw_line in resume_text.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 4))
        elif line.startswith("#"):
            heading = line.lstrip("# ").strip()
            if not title_added:
                story.append(Paragraph(html.escape(_pdf_safe_text(heading)), styles["ResumeTitle"]))
                title_added = True
            else:
                story.append(KeepTogether([
                    Paragraph(html.escape(_pdf_safe_text(heading)), styles["ResumeHeading"])
                ]))
        elif line.startswith(("- ", "* ", "• ")):
            body = html.escape(_pdf_safe_text(line[2:].strip())).replace("**", "")
            story.append(Paragraph(f"&bull; {body}", styles["ResumeBody"]))
        else:
            body = html.escape(_pdf_safe_text(line)).replace("**", "")
            story.append(Paragraph(body, styles["ResumeBody"]))

    if not story:
        raise ValueError("The generated resume has no content.")
    if not title_added:
        story.insert(0, Paragraph(html.escape(title), styles["ResumeTitle"]))

    SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title=f"Resume - {title}",
    ).build(story)
    return buffer.getvalue()


def _pdf_safe_text(text):
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")


st.set_page_config(page_title="RoleMatcherAI | Job Analyzer", page_icon="💼", layout="wide")
st.title("💼 RoleMatcherAI — Job Posting Analyzer")
st.write(
    "Paste a public job-posting URL to extract its job description, role matches, "
    "and technical skills. This analyzes the webpage only; it does not access "
    "browser history, cookies, or private browser data."
)

with st.form("job_url_form"):
    url = st.text_input("Job posting URL", placeholder="https://example.com/careers/software-engineer")
    uploaded_resume = st.file_uploader(
        "Optional: upload your current resume (PDF)",
        type=["pdf"],
        help="Used only to tailor the generated resume. Maximum 10 MB and 20 pages.",
    )
    submitted = st.form_submit_button("Analyze job posting", type="primary")

if submitted:
    st.session_state.pop("job_details", None)
    if not url.strip():
        st.error("Enter a job-posting URL.")
    else:
        try:
            page_bytes, page_encoding = fetch_page(url.strip())
            details = extract_job_details(page_bytes, page_encoding)
            if not details["description"]:
                raise PageFetchError("No job description text could be extracted from that page.")
            if uploaded_resume:
                resume_text = extract_resume_text(uploaded_resume.getvalue())
            else:
                resume_text = None
            st.session_state["job_details"] = details
            st.session_state["resume_text"] = resume_text
            st.session_state.pop("portfolio_match", None)
            st.session_state.pop("tailored_resume", None)
            try:
                st.session_state["portfolio_match"] = analyze_portfolio_match(details)
            except RuntimeError as error:
                st.error(str(error))
        except PageFetchError as error:
            st.error(str(error))
        except ValueError as error:
            st.error(str(error))
        except requests.RequestException as error:
            st.error(f"Could not retrieve the page: {error}")

details = st.session_state.get("job_details")
if details:
    st.subheader(details["title"])
    company_col, location_col, employment_col = st.columns(3)
    company_col.metric("Company", details["company"])
    location_col.metric("Location", details["location"])
    employment_col.metric("Employment type", details["employment_type"])

    st.subheader("Role matches")
    if details["roles"]:
        st.write(" · ".join(details["roles"]))
    else:
        st.info("No matching AI, software, or full-stack engineer role title was detected.")

    st.subheader("Technical skills")
    if details["skills"]:
        st.write(" · ".join(details["skills"]))
    else:
        st.info("No skills from the built-in technology list were detected.")

    if details["headings"]:
        st.subheader("Page headings")
        st.write(" · ".join(details["headings"]))

    st.subheader("Job description and page text")
    if details["structured_data_found"]:
        st.caption("Description extracted from structured JobPosting data.")
    st.text_area("Extracted text", details["description"], height=320, label_visibility="collapsed")

    match = st.session_state.get("portfolio_match")
    if match:
        st.subheader("Portfolio match")
        st.metric("Matched job skills", f"{match['skill_score']}%")
        if match["matched_skills"]:
            st.markdown("**Matching skills:** " + " · ".join(match["matched_skills"]))
        else:
            st.info("No listed job skills matched the indexed portfolio skills.")
        if match["missing_skills"]:
            st.markdown("**Job skills not found in the portfolio:** " + " · ".join(match["missing_skills"]))
        if match["evidence"]:
            with st.expander("Relevant portfolio evidence"):
                for item in match["evidence"]:
                    st.markdown(f"**{item['section']}**")
                    st.write(item["text"])

        if match["strong_match"]:
            st.success("Strong portfolio match. You can generate a tailored resume.")
            st.caption(
                "Generating a resume sends the job description, matching portfolio evidence, "
                "and optional uploaded resume text to Groq. Review the result for accuracy before using it."
            )
            if st.button("Create tailored resume", type="primary"):
                try:
                    with st.spinner("Generating a truthful, job-tailored resume with Groq..."):
                        tailored_resume = create_tailored_resume(
                            details,
                            match,
                            st.session_state.get("resume_text"),
                        )
                        st.session_state["tailored_resume"] = tailored_resume
                except RuntimeError as error:
                    st.error(f"Could not generate the resume: {error}")
                except ValueError as error:
                    st.error(f"Could not generate the resume: {error}")

            tailored_resume = st.session_state.get("tailored_resume")
            if tailored_resume:
                st.subheader("Review your tailored resume")
                st.text_area("Resume preview", tailored_resume, height=400)
                try:
                    pdf_bytes = resume_to_pdf(tailored_resume, details["title"])
                    safe_title = re.sub(r"[^A-Za-z0-9_-]+", "-", details["title"]).strip("-") or "job"
                    st.download_button(
                        "Download resume as PDF",
                        data=pdf_bytes,
                        file_name=f"tailored-resume-{safe_title}.pdf",
                        mime="application/pdf",
                    )
                except ValueError as error:
                    st.error(f"Could not create the resume PDF: {error}")
        else:
            st.warning(
                "This is not a strong portfolio match yet. Resume generation is available "
                f"when at least 3 listed skills match and the skill match is {STRONG_MATCH_THRESHOLD}% or higher."
            )
