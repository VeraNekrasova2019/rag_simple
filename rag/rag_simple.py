import time
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Constants
MODEL_NAME = "gpt-4o"
PDF_PATH = "../2nd-Edition-of-the-2017-code-2019.pdf"
PROMPT_TEMPLATE = """You are a cricket expert. You need to answer the question related to the law of cricket.
                     Given below is the context and question of the user.
                     context = {context}
                     question = {question}"""

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleRAG:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def build_simple_rag(self):
        """
        Build a SimpleRAG chain and invoke it with a sample question.
        """
        logger.info("Loading PDF document...")
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()

        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        logger.info("Creating embeddings...")
        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        vectorstore = self.create_vectorstore_with_rate_limiting(splits, embeddings)

        logger.info("Setting up prompt template...")
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        logger.info("Initializing language model...")
        llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0, api_key=self.api_key)
        retriever = vectorstore.as_retriever()

        rag_chain = (
                {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        logger.info("Invoking RAG chain with sample question...")
        return rag_chain.invoke("Tell me more about law number 30")

    @staticmethod
    def create_vectorstore_with_rate_limiting(splits, embeddings, max_retries=5, delay=1.0):
        """
        Create a vector store with rate limiting to handle API rate limits.
        """
        retries = 0
        while retries < max_retries:
            try:
                logger.info("Creating vector store...")
                vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
                return vectorstore
            except Exception as e:
                if "429" in str(e):
                    logger.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    retries += 1
                    delay *= 2  # Exponential backoff
                else:
                    logger.error("Failed to create vector store", exc_info=True)
                    raise e
        raise Exception("Failed to create vector store after multiple retries")

    @staticmethod
    def format_docs(docs):
        """
        Format documents into a single string with each document's content separated by double newlines.
        """
        return "\n\n".join(doc.page_content for doc in docs)