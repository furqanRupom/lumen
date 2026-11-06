from importlib.metadata import version

from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

print(f"LangChain OpenAI Version : {ChatOpenAI}")
print(f"LangChain Anthropic Version : {ChatAnthropic}")
print(version("langchain"))
print(version("langgraph"))


def main():

    # test openai
    llm_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm_openai.invoke("Say, Setup complete in one word")
    print(f"Response from OpenAI : {response}")

    # test claude
    llm_anthropic = ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929", temperature=0, timeout=30.99, stop=[]
    )
    response = llm_anthropic.invoke("Say, setup complte in one word")
    print(f"Respone from Claude : {response}")

    print("Hello First Rag System Application. We start here long way to go...")


if __name__ == "__main__":
    main()
