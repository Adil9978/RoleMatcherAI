"""Read Data/portfolio.txt and index its sections in a persistent ChromaDB."""

import re
from pathlib import Path

import chromadb


ROOT_DIR = Path(__file__).resolve().parent
PORTFOLIO_PATH = ROOT_DIR / "Data" / "portfolio.txt"
CHROMA_PATH = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "portfolio"
CHUNK_SIZE = 180
CHUNK_OVERLAP = 30


def split_sections(text):
    """Split Markdown portfolio text on level-one and level-two headings."""
    sections = []
    title = "Portfolio"
    lines = []

    for line in text.splitlines():
        heading = re.match(r"^#{1,2}\s+(.+?)\s*#*\s*$", line)
        if heading:
            content = "\n".join(lines).strip()
            if content:
                sections.append((title, content))
            title = heading.group(1).strip()
            lines = []
        else:
            lines.append(line)

    content = "\n".join(lines).strip()
    if content:
        sections.append((title, content))
    return sections


def make_chunks(text):
    """Create overlapping word chunks while keeping section metadata."""
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for section, content in split_sections(text):
        words = content.split()
        for start in range(0, len(words), step):
            chunk = " ".join(words[start:start + CHUNK_SIZE])
            if chunk:
                chunks.append((section, chunk))

    return chunks


def setup_chromadb():
    if not PORTFOLIO_PATH.is_file():
        raise FileNotFoundError(
            f"Portfolio file not found: {PORTFOLIO_PATH}\n"
            "Place portfolio.txt in the project's Data folder."
        )

    text = PORTFOLIO_PATH.read_text(encoding="utf-8")
    chunks = make_chunks(text)
    if not chunks:
        raise ValueError(f"No portfolio text found in {PORTFOLIO_PATH}")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"portfolio-{index:04d}" for index in range(len(chunks))]
    collection.upsert(
        ids=ids,
        documents=[chunk for _, chunk in chunks],
        metadatas=[
            {"source": PORTFOLIO_PATH.name, "section": section, "chunk_index": index}
            for index, (section, _) in enumerate(chunks)
        ],
    )

    existing_ids = set(collection.get()["ids"])
    stale_ids = existing_ids.difference(ids)
    if stale_ids:
        collection.delete(ids=sorted(stale_ids))

    print(f"Indexed {len(chunks)} portfolio chunks.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"ChromaDB directory: {CHROMA_PATH}")
    return collection


if __name__ == "__main__":
    setup_chromadb()