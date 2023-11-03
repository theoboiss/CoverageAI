"""A parser for code snippets.
This module offers to iterate over modules, maximizing the quantity of code per snippet and using any encoding method.
"""

import os
import sys
import inspect
import importlib.util
from abc import abstractmethod

import textwrap
import ast
import astor
from tiktoken import encoding_for_model


from blacklisted_unit_tests import BLACKLIST


class TooLongException(Exception):
    """Raised when a snippet is too long to be parsed."""


class _RoutineSnippet:
    """A snippet of routine code from a Python module.
    Attributes:
        routine_object: The routine object.
    """

    def __init__(self, routine_object):
        self.object = routine_object
        self.name = routine_object.__name__
        self.body = inspect.getsource(routine_object)

    def add_to_blacklist(self):
        """Adds the item to the blacklist."""

        BLACKLIST[self.get_snippet_type()][self.name] = None

    def is_blacklisted(self):
        """Checks if an item is blacklisted."""

        snippet_type = self.get_snippet_type()
        if snippet_type not in BLACKLIST:
            return False

        # Check if the item is blacklisted with keywords in the entire snippet type
        if None in BLACKLIST[snippet_type]:
            for keyword in BLACKLIST[snippet_type][None]:
                if keyword in self.body:
                    return True

        if self.name not in BLACKLIST[snippet_type]:
            return False

        # Check if the item is blacklisted without keyword
        if BLACKLIST[snippet_type][self.name] is None:
            return True

        # Check if the item is blacklisted with keywords
        for keyword in BLACKLIST[snippet_type][self.name]:
            if keyword in self.body:
                return True

        return False

    @classmethod
    def get_snippet_type(cls):
        """Gets the type of snippet.

        The type is the substring before the second uppercase letter.
        """

        class_name = cls.__name__[cls.__name__.rfind("_") + 1 :]  # Avoid the access modifier
        for i, char in enumerate(class_name[1:]):
            if char.isupper():
                return class_name[: 1 + i].lower()

    def __len__(self):
        return len(self.body)

    def __str__(self):
        return self.body


class _RoutineSnippetToEncode(_RoutineSnippet):
    """A snippet of routine code to encode from a Python module.
    Attributes:
        routine_object: The routine object.
        encode_func: The encoder function used to encode the snippet.
        max_len: The maximum length that the snippet can have.
    """

    def __init__(self, routine_object, encode_func, max_len):
        """Initializes the snippet."""

        super().__init__(routine_object)
        self.encode_func = encode_func
        self.max_len = max_len
        self.is_encodable = len(self) < self.max_len

    def _get_len(self, source_code):
        return len(self.encode_func(source_code))

    def __len__(self):
        return self._get_len(self.body)


class FunctionToEncode(_RoutineSnippetToEncode):
    """A snippet of a function from a Python module.
    Attributes:
        function_object_object: The function_object object.
        encode_func: The encoder function used to encode the snippet.
        max_len: The maximum length that the snippet can have.
    """

    def __init__(self, function_object, encode_func, max_len):
        super().__init__(function_object, encode_func, max_len)
        if self.is_encodable is False:
            raise TooLongException(f"`{self.name}` is longer than the maximum length ({self.max_len}).")
        if self.is_blacklisted():
            self.is_encodable = False
            raise NotImplementedError(f"`{self.name}` is blacklisted.")
        self.add_to_blacklist()


class _RegionToEncode:
    """A snippet of a region from a Python module.
    Attributes:
        encode_func: The encoder function used to encode the snippet.
        max_len: The maximum length that the snippet can have.
    """

    def __init__(self, functions_to_encode, encode_func, max_len):
        self.name = "[]"
        self.body = ""
        self.content = []
        self.encode_func = encode_func
        self.max_len = max_len
        self.is_encodable = len(self) < self.max_len
        self.surplus_functions = self._merge(functions_to_encode)

    def _merge(self, functions_to_encode):
        """Merges a list of functions to encode into a single region."""

        iterator_function_to_encode = iter(functions_to_encode)
        function_to_encode = next(iterator_function_to_encode, None)

        # Add the functions to encode to the region's content and body
        while function_to_encode is not None and (len(self) + self._get_len(function_to_encode.body)) < self.max_len:
            self.content.append(function_to_encode)
            self.body += function_to_encode.body + "\n"
            try:
                function_to_encode = next(iterator_function_to_encode, None)
            except StopIteration:
                break

        self.name = "[" + ", ".join(map(lambda x: x.name, self.content)) + "]"
        self.is_encodable = len(self) < self.max_len

        # If the loop stopped prematurely, return the remaining functions to encode
        remaining_functions_to_encode = []
        if function_to_encode is not None:
            remaining_functions_to_encode.append(function_to_encode)
            remaining_functions_to_encode.extend(iterator_function_to_encode)
        return remaining_functions_to_encode

    def _get_len(self, source_code):
        return len(self.encode_func(source_code))

    def __len__(self):
        return self._get_len(self.body)


class _ComponentToEncode(_RoutineSnippetToEncode):
    """A snippet of a component from a Python module.
    Attributes:
        component_object: The component object.
        encode_func: The encoder function used to encode its snippets.
        max_len: The maximum length that its snippets can have.
    """

    def __init__(self, component_object, encode_func, max_len):
        super().__init__(component_object, encode_func, max_len)
        self.doc = ""
        self.content = []

    def _build_encodable_functions(self, function_pair_name_object):
        """Builds an EncodableFunction from a pair (function_name, function_object)."""

        try:
            # There are in general at least 3 functions per component so we divide their maximum length by 3
            return FunctionToEncode(function_pair_name_object[1], self.encode_func, self.max_len // 3)
        except (NotImplementedError, TooLongException):
            return None

    @abstractmethod
    def _extend_content(self):
        """Extends the content of the component."""


class ClassToEncode(_ComponentToEncode):
    """A snippet of a class from a Python module.
    Attributes:
        class_object: The class object.
        encode_func: The encoder function used to encode the snippet.
        max_len: The maximum length that the snippet can have.
    """

    def __init__(self, class_object, encode_func, max_len):
        super().__init__(class_object, encode_func, max_len)
        if self.is_encodable:
            if self.is_blacklisted():
                self.is_encodable = False
                self._extend_content()
                if len(self.content) == 0:
                    raise NotImplementedError(f"{self.name} is blacklisted.")
        else:
            self._extend_content()
            if len(self.content) == 0:
                raise TooLongException(f"'{self.name}' is longer than the maximum length ({self.max_len}).")
        self.doc = _remove_function_bodies(self.body)
        self.add_to_blacklist()

    def _extend_content(self):
        """Extends the list of functions of the class."""

        # Get all the methods.
        methods_to_encode = filter(
            None,
            map(self._build_encodable_functions, inspect.getmembers(self.object, inspect.isfunction)),
        )

        # Merge them and add them as regions.
        while methods_to_encode:
            merged_functions_to_encode = _RegionToEncode(
                methods_to_encode, encode_func=self.encode_func, max_len=self.max_len
            )
            if len(merged_functions_to_encode.content) > 0:
                self.content.append(merged_functions_to_encode)
            methods_to_encode = merged_functions_to_encode.surplus_functions


class ModuleToEncode(_ComponentToEncode):
    """A snippet of a Python module.
    Attributes:
        module_object: The module object.
        encode_func: The encoder function used to encode the snippet.
        max_len: The maximum length that the snippet can have.
    """

    def __init__(self, module_object, encode_func, max_len):
        super().__init__(module_object, encode_func, max_len)
        if self.is_encodable:
            if self.is_blacklisted():
                self.is_encodable = False
                self._extend_content()
                if len(self.content) == 0:
                    raise NotImplementedError(f"{self.name} is blacklisted.")
        else:
            self._extend_content()
            if len(self.content) == 0:
                raise TooLongException(f"'{self.name}' is longer than the maximum length ({self.max_len}).")
        self.doc = _remove_function_bodies(self.body)
        self.add_to_blacklist()

    def _build_encodable_classes(self, class_pair_name_object):
        """Builds an EncodableClass from a class object."""

        try:
            return ClassToEncode(class_pair_name_object[1], self.encode_func, self.max_len)
        except TooLongException as too_long_exception:
            print(too_long_exception)
        except ValueError as value_error:
            print(value_error)
        return None

    def _extend_content(self):
        """Extends the list of classes and functions of the module."""

        # Add all the classes.
        self.content.extend(
            filter(None, map(self._build_encodable_classes, inspect.getmembers(self.object, inspect.isclass))),
        )

        # Get all the functions.
        functions_to_encode = filter(
            None,
            map(self._build_encodable_functions, inspect.getmembers(self.object, inspect.isfunction)),
        )

        # Merge them and add them as regions.
        while functions_to_encode:
            merged_functions_to_encode = _RegionToEncode(
                functions_to_encode, encode_func=self.encode_func, max_len=self.max_len
            )
            if len(merged_functions_to_encode.content) > 0:
                self.content.append(merged_functions_to_encode)
            functions_to_encode = merged_functions_to_encode.surplus_functions


def _remove_function_bodies(source_code, max_attempts=3):
    """Removes the bodies of all functions in the source code."""

    # Parse the source code into an abstract syntax tree (AST)
    attempts = 0
    while attempts < max_attempts:
        try:
            tree = ast.parse(source_code)
            break
        except IndentationError:
            # Fix indentation
            source_code = textwrap.dedent(source_code)
            attempts += 1
    else:
        return ""

    # Define a visitor class to modify the AST
    class BodyRemover(ast.NodeTransformer):
        """Removes the bodies of all functions in the AST.

        Note: Follows naming conventions of AST.
        """

        def visit_FunctionDef(self, node):
            """Removes the body of a function."""

            # Preserve the docstring
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                node.body = [node.body[0]]

            return node

        def visit_MethodDef(self, node):
            """Removes the body of a method."""

            # Preserve the docstring
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                node.body = [node.body[0]]

            return node

    # Create an instance of the visitor and apply it to the AST
    body_remover = BodyRemover()
    modified_tree = body_remover.visit(tree)

    # Convert the modified AST back to source code
    modified_source_code = astor.to_source(modified_tree)

    return modified_source_code


def _find_smallest_encodable_code_snippets(filepath, snippet_to_encode, context_of_snippet=""):
    """Iterates over the encodable code snippet.
    Args:
        snippet_to_encode: The code snippet to iterate over.
        doc_context: The documentation context.
    Yields:
        A tuple with the name of the code snippet and its content.
    Raises:
        ValueError: If the code snippet is not encodable.
    """

    if snippet_to_encode.is_encodable is True:
        yield filepath, snippet_to_encode.name, snippet_to_encode.body, context_of_snippet
    elif isinstance(snippet_to_encode, _ComponentToEncode):
        context_of_snippet = snippet_to_encode.doc
        for smaller_code_snippet in snippet_to_encode.content:
            yield from _find_smallest_encodable_code_snippets(filepath, smaller_code_snippet, context_of_snippet)
    else:
        raise ValueError(f"'{snippet_to_encode}' is not encodable.")


def _import_module_to_encode(filepath, encode_func, max_len):
    """Imports a Python module to encode.
    Args:
        filepath: The path to the module.
        encode_func: The encoder function used to encode the module.
        max_len: The maximum length that the module can have.
    Returns:
        The module to encode.
    """

    # Dynamically import a module from a file
    name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # May bug with front pages using Streamlit.
        sys.modules[name] = module

        # Parse the module to encode.
        return ModuleToEncode(module, encode_func, max_len)
    except (NotImplementedError, TooLongException) as no_unit_test_exception:
        print(no_unit_test_exception.__class__.__name__, ":", no_unit_test_exception)
    except (ValueError, ModuleNotFoundError) as source_code_exception:
        print(source_code_exception.__class__.__name__, ":", source_code_exception)
    except (KeyError, AttributeError) as probable_streamlit_error:
        print(probable_streamlit_error.__class__.__name__, "while importing", filepath, "due to Streamlit")
    return None


def iterate_encodable_code_snippets(source_code_directory, encode_func, max_len=1000):
    """Yields the name and content of each code snippet in the directory.
    Args:
        source_code_directory: The directory to inspect.
        max_len: The maximum length of the code content.
    Yields:
        A tuple with the name of the code snippet and its content.
    """

    original_directory = os.getcwd()
    os.chdir(source_code_directory)
    for root, _, files in os.walk("."):
        directory_name = os.path.basename(root)
        if directory_name in BLACKLIST["directory"] or any(
            keyword in root for keyword in BLACKLIST["directory"][None]
        ):
            continue
        for filename in files:
            if (
                not filename.endswith(".py")
                or filename in BLACKLIST["filename"]
                or any(keyword in filename for keyword in BLACKLIST["filename"][None])
            ):
                continue
            filepath = os.path.join(root, filename)
            module_to_encode = _import_module_to_encode(filepath, encode_func, max_len)
            if module_to_encode is not None:
                yield from _find_smallest_encodable_code_snippets(filepath, module_to_encode)
    os.chdir(original_directory)
    with open("blacklisted_unit_tests.py", mode="w", encoding="utf-8") as blacklist:
        blacklist.writelines(
            [
                "# Blacklisted items will not be iterated if they contain the specified substrings/keywords.\n",
                "# If a substring list is set to None, the item is blacklisted.\n",
                "# All substrings associated with a None element are associated with all elements of the same type.\n",
                "# The `routine` type includes methods, functions and decorators.\n",
                "BLACKLIST = " + str(BLACKLIST),
            ]
        )
    print("Updated blacklist in blacklisted_unit_tests.py")


# Demo
# For testing only
if __name__ == "__main__":
    directory_path = "."
    for snippet_filepath, snippet_name, snippet_body, snippet_context in iterate_encodable_code_snippets(
        directory_path, encoding_for_model("gpt-3.5-turbo").encode, max_len=2600
    ):
        print("Filepath:", snippet_filepath)
        print("Name:", snippet_name)
        # print("Content:")
        # print(snippet_body)
        # print("Context:")
        # print(snippet_context)
