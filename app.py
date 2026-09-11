from rag import ask_question


print("===================================")
print("     Document Q&A AI Assistant")
print("===================================")

print("Type 'exit' to close the application.")


while True:

    question = input("\nEnter your question: ")

    if question.lower() == "exit":
        print("Application closed.")
        break

    answer = ask_question(question)

    print("\nAI Answer:")
    print(answer)
