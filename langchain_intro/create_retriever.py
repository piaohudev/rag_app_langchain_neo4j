import dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

dotenv.load_dotenv()

loader = CSVLoader(
    file_path=REVIEWS_CSV_PATH,
    source_column="review",
)
reviews = loader.load()

reviews_vector_db = Chroma.from_documents(
    reviews, OpenAIEmbeddings(), persist_directory=REVIEWS_CHROMA_PATH
)

test_reviews_vector_db = Chroma(
    persist_directory="chroma_data/",
    embedding_function=OpenAIEmbeddings(),
)

if __name__ == '__main__':
    question = """
        Has anyone complained about communication with the hospital staff?
    """

    relevant_docs = test_reviews_vector_db.similarity_search(question, k=3)

    response = ""
    for doc in relevant_docs:
        response += f"{doc.page_content}\n================================\n"

    print(response)
