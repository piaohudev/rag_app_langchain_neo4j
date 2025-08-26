import dotenv
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

if __name__ == '__main__':
    response = chat_model.invoke("What is blood pressure?")
    print(response.content)
