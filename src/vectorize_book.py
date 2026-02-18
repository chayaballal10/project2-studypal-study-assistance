import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CLASS_SUBJECT_NAME = os.getenv('CLASS_SUBJECT_NAME')
DEVICE = os.getenv('DEVICE', 'cpu')

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

data_dir = os.path.join(parent_dir, "data")
vector_db_dir = os.path.join(parent_dir, "vector_db")
chapters_vector_db_dir = os.path.join(parent_dir, "chapters_vector_db")


# ---------------- EMBEDDINGS ----------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": DEVICE}
)

# ---------------- TEXT SPLITTER ----------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=["\n\n", "\n", ". ", " ", ""]
)


# ---------------- HELPERS ----------------
def clean_old_db(path):
    """Delete old DB to avoid duplicate embeddings"""
    if os.path.exists(path):
        print(f"Deleting old DB: {path}")
        shutil.rmtree(path)


def add_metadata(documents, source_name):
    """Add page + chapter metadata"""
    for doc in documents:
        doc.metadata["source"] = source_name
        if "page" not in doc.metadata:
            doc.metadata["page"] = 0
    return documents


# ---------------- BOOK VECTORIZE ----------------
def vectorize_book_and_store_to_db(class_subject_name, vector_db_name):
    book_dir = os.path.join(data_dir, class_subject_name)
    vector_db_path = os.path.join(vector_db_dir, vector_db_name)

    if not os.path.exists(book_dir):
        print("Book directory does not exist!")
        return

    clean_old_db(vector_db_path)

    loader = DirectoryLoader(
        path=book_dir,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader
    )

    documents = loader.load()

    if not documents:
        print("No documents found.")
        return

    documents = add_metadata(documents, class_subject_name)

    text_chunks = text_splitter.split_documents(documents)

    Chroma.from_documents(
        documents=text_chunks,
        embedding=embedding,
        persist_directory=vector_db_path
    )

    print(f"✅ {class_subject_name} saved to vector db: {vector_db_name}")


# ---------------- CHAPTER VECTORIZE ----------------
def vectorize_chapters(class_subject_name):
    book_dir = os.path.join(data_dir, class_subject_name)

    if not os.path.exists(book_dir):
        print("Book directory does not exist!")
        return

    for chapter in os.listdir(book_dir):
        if not chapter.endswith('.pdf'):
            continue

        chapter_name = chapter[:-4]
        chapter_pdf_path = os.path.join(book_dir, chapter)
        chapter_db_path = os.path.join(chapters_vector_db_dir, chapter_name)

        clean_old_db(chapter_db_path)

        loader = PyMuPDFLoader(chapter_pdf_path)
        documents = loader.load()

        if not documents:
            print(f"Skipping {chapter_name} (no content)")
            continue

        documents = add_metadata(documents, chapter_name)

        texts = text_splitter.split_documents(documents)

        Chroma.from_documents(
            documents=texts,
            embedding=embedding,
            persist_directory=chapter_db_path
        )

        print(f"✅ {chapter_name} chapter vectorized")
