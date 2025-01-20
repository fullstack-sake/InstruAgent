import re
from typing import Union

from termcolor import colored

from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.agent import AgentOutputParser
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain_core.exceptions import OutputParserException

FINAL_ANSWER_ACTION = "Final Answer:"
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action:' after 'Thought:'"
)
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action Input:' after 'Action:'"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)

USER_MESSAGE = (
    "Please regenerate the plan, user's correction request is:"
)


class ReActSingleInputOutputParser(AgentOutputParser):

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        pattern = r'^Final Answer: .*?\.$'
        match = re.search(pattern, text, flags=re.MULTILINE)
        regex = r"Action\s*\d*\s*:\s*(.*?)\s*Action\s*\d*\s*Input\s*\d*\s*:([^$\n]*)"

        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            new_text = text
            if match:
                new_text = re.sub(pattern, '', text, flags=re.MULTILINE)

            action = action_match.group(1).strip()
            action_input = action_match.group(2).strip()
            action_input = action_input.strip('"')
            corrected_input = input(colored(f"\n是否执行:{action}, {action_input}\n", "red")).strip()
            bool_map = {"是": True, "否": False, '\n': True, 'yes': True, '': True}
            bool_value = bool_map.get(corrected_input, False)
            if bool_value:
                return AgentAction(action, action_input, new_text)
            else:
                user_input = input(colored(f"更改需求为：", "red")).strip()+f'\n'
                raise OutputParserException(
                    USER_MESSAGE+user_input,
                    observation="",
                    llm_output='output'+new_text,
                    send_to_llm=True,
                )

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

    @property
    def _type(self) -> str:
        return "react-single-input"
