import pickle
from query_data import get_chain


if __name__ == "__main__":
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    qa_chain = get_chain(vectorstore)
    print("Chat with your Data!")
    while True:
        print("Human:")
        question = input()
        result = qa_chain({"question": question, "chat_history": []})
        print("AI:")
        print(result["answer"])
