from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based on the provided context.

If you cannot answer the question based on the context, say "I don't have enough information to answer that question."

Do not make up information. Only use the context provided.

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


class RAGChain:

    def __init__(self, vector_store_service: VectorStoreService | None = None):
      
        self.vector_store = vector_store_service or VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self._evaluator = None

        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openai_api_key,
        )

        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info(
            f"RAGChain initialized with model={settings.llm_model}, "
            f"retrieval_k={settings.retrieval_k}"
        )

    @property
    def evaluator(self):
        if self._evaluator is None:
            from app.core.ragas_evaluator import RAGASEvaluator

            self._evaluator = RAGASEvaluator()
        return self._evaluator

    def query(self, question: str) -> str:
    
        logger.info(f"Processing query: {question[:100]}...")

        try:
            answer = self.chain.invoke(question)
            logger.info("Query processed successfully")
            return answer
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def query_with_sources(self, question: str) -> dict:
        
        logger.info(f"Processing query with sources: {question[:100]}...")

        try:
            answer = self.chain.invoke(question)

            source_docs = self.retriever.invoke(question)

            sources = [
                {
                    "content": (
                        doc.page_content[:500] + "..."
                        if len(doc.page_content) > 500
                        else doc.page_content
                    ),
                    "metadata": doc.metadata,
                }
                for doc in source_docs
            ]

            logger.info(f"Query processed with {len(sources)} sources")

            return {
                "answer": answer,
                "sources": sources,
            }
        except Exception as e:
            logger.error(f"Error processing query with sources: {e}")
            raise

    async def aquery(self, question: str) -> str:
       
        logger.info(f"Processing async query: {question[:100]}...")

        try:
            answer = await self.chain.ainvoke(question)
            logger.info("Async query processed successfully")
            return answer
        except Exception as e:
            logger.error(f"Error processing async query: {e}")
            raise

    async def aquery_with_sources(self, question: str) -> dict:
       
        logger.info(f"Processing async query with sources: {question[:100]}...")

        try:
            
            answer = await self.chain.ainvoke(question)

            source_docs = self.retriever.invoke(question)

            sources = [
                {
                    "content": (
                        doc.page_content[:500] + "..."
                        if len(doc.page_content) > 500
                        else doc.page_content
                    ),
                    "metadata": doc.metadata,
                }
                for doc in source_docs
            ]

            logger.info(f"Async query processed with {len(sources)} sources")

            return {
                "answer": answer,
                "sources": sources,
            }
        except Exception as e:
            logger.error(f"Error processing async query with sources: {e}")
            raise

    async def aquery_with_evaluation(self, question: str, include_sources: bool = True) -> dict:
        logger.info(f"Processing query with evaluation: {question[:100]}...")

        try: 
            result = await self.aquery_with_sources(question)
            answer = result["answer"]
            sources = result["sources"]

            contexts = [source["content"] for source in sources]

            try:
                evaluation = await self.evaluator.aevaluate(
                    question=question, answer=answer, contexts=contexts
                )
                logger.info(
                    f"Evaluation completed - "
                    f"faithfulness={evaluation.get('faithfulness', 'N/A')}, "
                    f"answer_relevancy={evaluation.get('answer_relevancy', 'N/A')}"
                )
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}", exc_info=True)
                evaluation = {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "evaluation_time_ms": None,
                    "error": str(e),
                }

            return {"answer": answer, "sources": sources, "evaluation": evaluation}

        except Exception as e:
            logger.error(f"Error in query with evaluation: {e}")
            raise

    def stream(self, question: str):
        logger.info(f"Streaming query: {question[:100]}...")

        try:
            for chunk in self.chain.stream(question):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming query: {e}")
            raise