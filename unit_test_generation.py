"""Command line tool for generating test code from a chat with GPT."""

import glob
import json
import os
import re
import subprocess

import openai
from tiktoken import encoding_for_model

from code_snippet_parser import iterate_encodable_code_snippets


CODE_DIRECTORY = "."
AUTOMATIC = True  # Fully automatic generation if True, otherwise semi-automatic with user confirmation.
CONTEXT = 16000  # tokens
TEMPERATURE = 0.6  # 0.0-0.4: focused, conservative. 0.5-0.7: between coherence and creativity. 0.8-1.4: exploratory.

openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = {4000: "gpt-3.5-turbo", 16000: "gpt-3.5-turbo-16k"}[CONTEXT]
PRICING = {4000: 0.0015, 16000: 0.003}[CONTEXT]
ENCODER = encoding_for_model(MODEL)


def len_encoding(string):
    """Return the length of the string in tokens."""

    return len(ENCODER.encode(string))


class Context:
    """A context for the chat with the LLM.

    The Context class represents a context for a conversation with the LLM.
    It keeps track of a set of messages and the maximum number of tokens that
    can be used for generating replies.

    Chat messages take the following form:
    [knowledge_prompt, code_snippet_context, system_prompt, code_snippet_to_test].

    The code_snippet_context is the source code file stripped of all function definitions.
    It contains only the names of global variables, classes, methods, functions and their documentation.

    Attributes:
        max_tokens (int): The maximum number of tokens that can be used by the context.
    """

    def __init__(self, max_tokens) -> None:
        self._max_tokens = max_tokens
        self._system_prompt = [create_test_cases_prompt()]
        self._knowledge_prompt = []
        self._user_prompts = []
        self._token_len_messages = []
        self._token_counter = 0
        self._deletion_counter = 0

    @property
    def max_tokens(self):
        """Return the maximum number of tokens in the context."""

        return self._max_tokens

    @property
    def messages(self):
        """Return the messages in the context."""

        if len(self._user_prompts) == 0:
            return self._knowledge_prompt + self._system_prompt
        return self._knowledge_prompt + self._user_prompts[:-1] + self._system_prompt + [self._user_prompts[-1]]

    @property
    def total_tokens(self):
        """Return the total number of tokens in the context."""

        return self._token_counter

    @property
    def messages_deleted(self):
        """Return the number of messages deleted from the context."""

        return self._deletion_counter

    @property
    def roles(self):
        """Return the roles of the messages in the context."""

        return list(set(map(lambda message: message["role"], self.messages)))

    @property
    def contents(self):
        """Return the contents of the messages in the context."""

        return list(map(lambda message: message["content"], self.messages))

    def setup(self, code_snippet_context):
        """Adapt the knowledge of the chatbot."""

        if len(code_snippet_context) > 0:
            self._knowledge_prompt = [create_knowledge_prompt(code_snippet_context)]
        self.clean()

    def add(self, role, content):
        """Add a message to the context.

        The message will be added to the end of the context. If the context would
        overflow, the oldest messages will be forgotten.

        Args:
            role: The role of the message.
            content: The content of the message.
        """

        num_tokens_content = self.__class__.get_token_length({"role": role, "content": content})
        if num_tokens_content > self._max_tokens:
            raise ValueError(
                f"The new content ({num_tokens_content} tokens) would overload the context "
                f"(max. {self._max_tokens} tokens)."
            )
        self._user_prompts.append({"role": role, "content": content})

    def update(self, tokens_prompt, tokens_completion):
        """Update the count of tokens used in the context and clean it after the chatbot replied.

        The number of tokens used in the context is updated by adding the number
        of tokens from prompt and completion. If the context would overflow,
        the oldest messages will be forgotten.

        Args:
            number_tokens_prompt: The number of tokens from prompt.
            number_tokens_completion: The number of tokens from completion.
        """

        self._token_len_messages = tokens_prompt
        self._token_counter += tokens_prompt
        self._token_counter += tokens_completion
        self.clean()

    def clean(self, remove=-1):
        """Clear as many messages as necessary to fit the context."""

        self._clean(desired_number_of_tokens=(len(self) - remove) if remove != -1 else self._max_tokens)

    def clear(self):
        """Clear all messages from the context."""

        self._user_prompts = []
        self._token_len_messages = 0

    @staticmethod
    def get_token_length(message):
        """Return the length of the message in tokens."""

        # Every message follows '<im_start>{role}\n{content}<im_end>\n'
        return len_encoding(message["role"]) + len_encoding(message["content"]) + 4

    def _clean(self, desired_number_of_tokens):
        if desired_number_of_tokens < 1:
            return

        while desired_number_of_tokens < len(self):
            self._user_prompts.pop(0)  # Forget old context
            self._deletion_counter += 1

    def _get_tokens_length_from_messages(self):
        """Returns the number of tokens used by a list of messages."""

        num_tokens = 0
        for message in self.messages:
            # Every message follows '<im_start>{role}\n{content}<im_end>\n' so use get_token_length
            num_tokens += self.get_token_length(message)
        return num_tokens

    def __len__(self):
        return self._get_tokens_length_from_messages()


def write_unit_test(source_code_filepath: str, unit_test: str, mode: str = "a"):
    """A function that writes a test file for the code snippet.

    Args:
        source_code_filepath (str): The path to the code snippet.
        unit_test (str): The unit test markdown containing the code to write.
        mode (str, optional): The mode to open the file in. Defaults to "a".

    Returns:
        str: The path of the test file.
    """

    source_code_directory, module_name = os.path.split(source_code_filepath)
    test_directory = _get_test_directory(source_code_directory, "unit")
    test_filepath = os.path.join(test_directory, f"test_{module_name}")

    unit_test_code = _extract_code_from_markdown(unit_test)
    if not os.path.isfile(test_filepath):
        imports = (
            f"import pytest\n\nfrom {source_code_filepath.replace('/', '.')[2:].replace('.py', '')} import *\n\n"
        )
        unit_test_code = imports + unit_test_code
    else:
        unit_test_code = "\n\n" + unit_test_code

    with open(test_filepath, mode, encoding="utf-8") as test_file:
        test_file.write(unit_test_code)

    return test_filepath


def generation_analysis(finished_context, processed_snippets_filepaths, encountered_snippets_anomalies):
    """A function that prints out metrics of the generation process.

    Args:
        finished_context (Context): The context of the generation process.
        processed_snippets_filepaths (dict): The source code filepaths that have been processed.
        encountered_snippets_anomalies (list): The source code anomalies that have been encountered.
    """

    print()
    print("###################")
    print("GENERATION ANALYSIS")
    print()

    # Source code anomalies

    print(
        f"{len(encountered_snippets_anomalies)} files contained anomalies"
        f"{':' if len(encountered_snippets_anomalies) else ''}"
    )
    print(", ".join(encountered_snippets_anomalies))
    print()

    # Test files generated

    number_snippets = sum(map(len, processed_snippets_filepaths.values()))

    print(f"{len(processed_snippets_filepaths)} source code files processed")
    try:
        print(f"{round(number_snippets / len(processed_snippets_filepaths), 2)} source code snippets per file")
    except ZeroDivisionError:
        print("0 source code snippets per file")
    sum_tokens = sum(map(sum, processed_snippets_filepaths.values()))
    print(f"{round(sum_tokens / number_snippets, 2)} tokens per source code snippet")
    print(f"{number_snippets} source code snippets has been processed")
    print(f"{sum_tokens} tokens in total in the source code")
    print()

    # Context

    print(f"{finished_context.messages_deleted} context clean-ups in file encountered")
    try:
        print(
            f"{round(number_snippets / (finished_context.messages_deleted))}"
            " source code snippets processed per context status (between clean-ups) in file"
        )
    except ZeroDivisionError:
        print("All the source code snippets have been processed without clean-up via different files")
    print(f"{finished_context.total_tokens} tokens have been exchanged with the API")
    print(f"${round(finished_context.total_tokens / 1000 * PRICING, 2)} in total")
    print()


def create_knowledge_prompt(code_snippets_context):
    """A function returns a prompt for the chatbot to use so he knows the code snippets context.

    Args:
        code_snippets_context (str): The code snippets context.

    Returns:
        A result string from create chat completion. Knowledge for the submitted code in response.
    """

    return {
        "role": "system",
        "content": f"You have knowledge of the following source code: ```python\n{code_snippets_context}```",
    }


def create_test_cases_prompt():
    """A function that takes in code and focus topics and returns a prompt for the chatbot to use.

    Returns:
        A result string from create chat completion. Test cases for the submitted code in response.
    """

    function_string = "def get_pytest_unit_tests_str(code: str) -> str:"
    description_string = "Generate unit tests with pytest for the submitted code."

    return _create_func_prompt(function_string, description_string)


def _create_func_prompt(function: str, description: str):
    """A function that builds a prompt for the chatbot to use.

    Args:
        function (str): The function to call
        description (str): The description of the function

    Returns:
        list: The next messages to send to the chatbot.
    """

    return {
        "role": "system",
        "content": f"You are now the following python function: ```# {description}"
        f"\n{function}```\n\nOnly respond with your `return` value.",
    }


def _extract_code_from_markdown(markdown: str):
    """A function that extracts code from markdown.

    Args:
        markdown (str): The markdown to extract the code from.

    Returns:
        str: The code from the markdown.
    """

    default_function_string = "def get_pytest_unit_tests_str(code: str) -> str:\n    "
    default_return_string = [
        "return ",
        "code = ",
        "tests = ",
        "unit_tests = ",
        "source_code = ",
        "test_code = ",
        "test = ",
    ]
    default_string_separator = ['"""', "'''"]
    default_import_string = ["from your_module", "import your_module", "from module", "import module"]

    markdown_head = markdown[: min(len(markdown), 500)]

    # Remove markdown formatting of code blocks
    if "```" in markdown_head:
        markdown = markdown.split("```")[1]
        if markdown.startswith("python\n"):
            markdown = markdown.split("python\n")[1]
        if markdown.startswith("pytest\n"):
            markdown = markdown.split("pytest\n")[1]

    # Remove the default function string (role taken by the LLM model)
    if default_function_string in markdown_head:
        markdown = markdown.split(default_function_string)[1]
        for return_string in default_return_string:
            if markdown.startswith(return_string):
                markdown = markdown.split(return_string)[1]
        for string_separator in default_string_separator:
            if markdown.startswith(string_separator):
                markdown = markdown.split(string_separator)[1]

    # Remove the entire line containing "from your_module"
    for import_string in default_import_string:
        if import_string in markdown_head:
            markdown_lines = markdown.split("\n")
            markdown_lines = [line for line in markdown_lines if not line.startswith(import_string)]
            markdown = "\n".join(markdown_lines)

    return markdown


def _create_init_files(path: str):
    """Create __init__.py files in each folder of the given path.

    Args:
        directory (str): The directory to create the init file in.
    """

    # Avoid creating __init__.py files in the current directory
    if path.startswith("." + os.path.sep):
        path = path[2:]

    current_dir = ""
    for directory in path.split(os.path.sep):
        current_dir = os.path.join(current_dir, directory)
        init_filepath = os.path.join(current_dir, "__init__.py")
        if not os.path.exists(init_filepath):
            with open(init_filepath, "a", encoding="utf-8"):
                pass


def _get_test_directory(source_code_directory: str, test_type: str):
    """Get the test directory path based on the source code directory.

    Args:
        source_code_directory (str): The path to the source code directory.
        test_type (str, optional): The type of the test directory. Can be "integration" or "unit".

    Returns:
        str: The absolute path of the test directory.
    """

    source_code_relative_path = os.path.relpath(source_code_directory, os.getcwd())

    test_directory = os.path.join("tests", test_type, source_code_relative_path)
    os.makedirs(test_directory, exist_ok=True)

    _create_init_files(test_directory)

    return test_directory


if __name__ == "__main__":
    snippets_filepaths = {}  # Register the number of tockens of each snippet for each file
    snippets_anomalies = []  # Register the empty snippets
    context = Context(max_tokens=CONTEXT)
    try:
        for snippet_filepath, snippet_name, snippet_body, snippet_context in iterate_encodable_code_snippets(
            source_code_directory=CODE_DIRECTORY,
            encode_func=ENCODER.encode,
            max_len=(CONTEXT // 6 - len_encoding("user") - 4),
        ):
            if not snippet_body:
                snippets_anomalies.append(snippet_filepath)
                print(f"Anomaly: {snippet_filepath} has no body")
                continue
            
            print(f"Creating test cases for {snippet_filepath}")
            if snippet_filepath not in snippets_filepaths:
                snippets_filepaths.setdefault(snippet_filepath, [])
                context.clear()
            snippets_filepaths[snippet_filepath].append(len_encoding(snippet_body))

            context.summary = snippet_context
            context.add(role="user", content=snippet_body)

            # Allow the user to skip the test case creation.
            print(snippet_body)
            if not input(f"Do  you want to create test cases for {snippet_name}? (y/N): ").lower() == "y":
                print()
                print("###################")
                print()
                continue

            # Prompt the LLM with the context and the source code.
            print("Waiting for the LLM to respond...")
            chat = openai.ChatCompletion.create(model=MODEL, temperature=TEMPERATURE, messages=context.messages)
            print()
            context.set_prompt_token_length(int(chat.usage.prompt_tokens))
            context.set_completion_token_length(int(chat.usage.completion_tokens))
            reply = chat.choices[0].message.content

            write_unit_test(snippet_filepath, reply, "a")
    finally:
        generation_analysis(context, snippets_filepaths, snippets_anomalies)
