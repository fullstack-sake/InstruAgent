import pickle
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import Tool


from util import LimitedHistoryMemory

from tools import *
from fb_planning import *
import importlib.util
import sys

##
from llm_core import *
##


question_template = """
你现在扮演一名电子测量工程师，用户的需求是：“{input}”。
你现在身边有示波器和信号发生器，请你按步骤规划每一步需要如何进行测量，你只需要简略说明每一步要做什么，尽量步骤少，你需要给我确切的仪器测试要设置的数据（比如幅度、频率 etc.），这些数据是输入到真实仪器的。
这是一个示例模板：
1.初始化仪器（根据需求），设置CHAN1通道的显示状态ON、耦合方式和电阻值。
2.调整示波器：将示波器设置为DC耦合模式。
3.设置信号发生器（非必须，不需要就不使用）：设置方波信号，频率为1kHz，幅度为5V。
4...
"""

def exception_handler(func):
    """Decorator for handling exceptions and printing error messages."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except AttributeError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except NameError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            return func(*args, **kwargs)
            # sys.exit(1)
    return wrapper

@exception_handler
def load_templates(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        return data["templates"]


# def _handle_error(error) -> str:
#     # if 
#     print("错误：正在重启")
#     main()
#     return str(error)
# Define a global variable to count the number of restarts
restart_count = 0

def _handle_error(error) -> str:
    global restart_count
    
    # if restart_count < 2:
    #     print(f"错误：正在重启，当前{restart_count}")
    #     restart_count += 1  # Increment the counter if an error occurs
    #     main()  # Attempt to restart the main function

    # else:
    #     print("重启次数已达上限，系统无法恢复。")
    
    return str(error)

@exception_handler
def main():
    templates = load_templates('templates_data22.pkl')
    llm = LLM()
    memory = LimitedHistoryMemory(memory_key="chat_history", return_messages=True, max_history_length=10)
    output_parser = StrOutputParser()
    prompt_process = ChatPromptTemplate.from_template(question_template)
    prompt_agent = ChatPromptTemplate.from_template(templates)
    process_chain = prompt_process | llm | output_parser | {"process": RunnablePassthrough()}
    agent = create_react_agent(llm, tools, prompt=prompt_agent)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=_handle_error,max_execution_time=30)


    total_chain = process_chain | agent_executor
    print(total_chain.get_graph().print_ascii())
    response1 = total_chain.invoke({"input": "帮我测试一下这个运算放大器的压摆率"})
    print(str(response1))
    print("历史记录:", memory.load_memory_variables({})["chat_history"])


if __name__ == "__main__":
    main()
