import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


# Embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# Load Chroma database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# OpenAI model
llm = ChatOpenAI(
    model="gpt-5.6-luna"
)


# Prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful document assistant.

Answer the question using ONLY the provided context.

If the answer is not present in the context,
say: "I could not find the answer in the document."

Context:
{context}

Question:
{question}
""")


def ask_question(question):

    # Retrieve relevant chunks
    documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Create prompt
    messages = prompt.invoke({
        "context": context,
        "question": question
    })

    # Ask OpenAI
    response = llm.invoke(messages)

    return response.content


if __name__ == "__main__":

    while True:

        question = input("\nAsk a question: ")

        if question.lower() == "exit":
            break

        answer = ask_question(question)

        print("\nAnswer:")
        print(answer)