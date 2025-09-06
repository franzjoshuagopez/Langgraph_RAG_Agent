import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from apps.chat.rag.agent import init_rag, run_agent

def main():

    init_rag()

    user_input = "What is the performance of the stock market?"

    response = run_agent(user_input)

    print("user: ", user_input)
    print("agent: ", response)

if __name__ == "__main__":
    main()