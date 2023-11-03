# Blacklisted items will not be iterated if they contain the specified substrings/keywords.
# If a substring list is set to None, the item is blacklisted.
# All substrings associated with a None element are associated with all elements of the same type.
# The `routine` type includes methods, functions and decorators.
BLACKLIST = {
    "directory": {
        None: {"test", "__"},
    },
    "filename": {
        None: ["__"],
        "blacklisted_unit_tests.py": None,
        "code_snippet_parser.py": None,
        "unit_test_generation.py": None,
    },
    "module": {
        None: ["streamlit.", " st."],
    },
    "class": {
        None: ["streamlit.", " st."],
    },
    "function": {
        None: ["streamlit.", " st."],
        "abstractmethod": None,
        "__init__": None,
    },
}
