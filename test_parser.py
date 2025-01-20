import re
from typing import Union

# 定义必要的类和异常
class AgentAction:
    def __init__(self, action, action_input, text):
        self.action = action
        self.action_input = action_input
        self.text = text

class AgentFinish:
    def __init__(self, output, text):
        self.output = output
        self.text = text

class OutputParserException(Exception):
    def __init__(self, message, observation=None, llm_output=None, send_to_llm=False):
        super().__init__(message)
        self.observation = observation
        self.llm_output = llm_output
        self.send_to_llm = send_to_llm

# 定义常量
FINAL_ANSWER_ACTION = "Final Answer:"
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = "Final answer and parsable action found."
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = "Missing action after thought."
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = "Missing action input after action."

class Parser:
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # includes_answer = FINAL_ANSWER_ACTION in text
        pattern = r'^Final Answer: .*?\.$'
        match = re.search(pattern, text, flags=re.MULTILINE)
        regex = r"Action\s*\d*\s*:\s*(.*?)\s*Action\s*\d*\s*Input\s*\d*\s*:([^$\n]*)"

        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            new_text = text
            if match:
                new_text = re.sub(pattern, '', text, flags=re.MULTILINE)
                # raise OutputParserException(
                #     f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
                # )
                # return AgentFinish(
                #     {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
                # )
            
            action = action_match.group(1).strip()
            action_input = action_match.group(2).strip()
            action_input = action_input.strip('"')
            corrected_input = input(f"是否执行:{action}, {action_input}").strip()
            bool_map = {"是": True, "否": False}
            bool_value = bool_map.get(corrected_input, False)
            # action_input = corrected_input
            # print(new_text)
            # return AgentAction(action, action_input, text)
            if bool_value:
                return AgentAction(action, action_input, new_text)

        elif match:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        if not re.search(r"Action\s*\d*\s*:\s*(.*?)\s*(.*?)$", text, re.MULTILINE):
            raise OutputParserException(
                f"Check your output and make sure it conforms, use the Action/Action Input syntax",
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(
            r"\s*Action\s*\d*\s*Input\s*\d*\s*:\s*(.*?)\s*(.*?)$", text, re.MULTILINE
        ):
            raise OutputParserException(
                f"Check your output and make sure it conforms, use the Action/Action Input syntax",
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        else:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")





# 创建 Parser 实例
parser = Parser()

# 示例 1: 解析动作和输入
text_with_action = """
The oscilloscope channel has been set up. Now, I need to observe the waveform and calculate the rise and fall times to determine the slew rate.

Action:1231231

Action Input: 12312

示波器上的波形已显示给用户
I have successfully observed the waveform on the oscilloscope. Now, I need to calculate the rise and fall times to determine the slew rate.
I have successfully calculated the slew rate. Now, I can proceed to the final answer.

Final Answer: The slew rate of the waveform is 0.0000 V/us.
"""
try:
    result = parser.parse(text_with_action)
    if isinstance(result, AgentAction):
        print(f"Action111: {result.action}===")
        print(f"Action Input111: {result.action_input}===")
        # print(f"=========================Original Text: {result.text}")
except OutputParserException as e:
    print(f"Exception: {e}")

# # 示例 2: 解析最终答案
# text_with_final_answer = """

# Action:1.abc
# 2.abc
# 3.abc
# 4.abc
# Action Input: 12312

# """
# try:
#     result = parser.parse(text_with_final_answer)
#     if isinstance(result, AgentFinish):
#         print(f"Output: {result.output}")
#         print(f"Original Text: {result.text}")
# except OutputParserException as e:
#     print(f"Exception: {e}")

# # 示例 3: 无效格式的文本
# invalid_text = "This is an invalid format"
# try:
#     result = parser.parse(invalid_text)
# except OutputParserException as e:
#     print(f"Exception: {e}")
