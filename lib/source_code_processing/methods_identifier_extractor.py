import os
import re
import pandas as pd
import numpy as np

"""common_keywords = {
    # Common C++ keywords
    'include', 'h', 'hpp', 'true', 'false', 'void', 'const', 'int', 'float', 'double', 'char', 'bool', 'static',
    'return', 'if', 'else', 'while', 'for', 'switch', 'case', 'break', 'continue', 'typedef', 'class', 'struct',
    'namespace', 'using', 'std', 'iostream', 'vector', 'string', 'map', 'set', 'algorithm',
    
    # Common Java keywords
    'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const', 'continue', 'default',
    'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements', 'import',
    'instanceof', 'int', 'interface', 'long', 'native', 'new', 'null', 'package', 'private', 'protected', 'public', 
    'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 
    'try', 'void', 'volatile', 'while', 
    
    # Common types and libraries
    'String', 'Integer', 'System', 'out', 'println', 'args', 'List', 'ArrayList', 'HashMap', 'Set', 'HashSet'
}

common_keywords = {
    # C/C++ Keywords
    'include', 'h', 'hpp', 'true', 'false', 'void', 'const', 'int', 'float', 'double', 'char', 'bool', 'static',
    'return', 'if', 'else', 'while', 'for', 'switch', 'case', 'break', 'continue', 'typedef', 'class', 'struct',
    'namespace', 'using', 'std', 'iostream', 'vector', 'string', 'map', 'set', 'algorithm', 'inline', 'extern',
    'friend', 'volatile', 'explicit', 'virtual', 'long', 'short', 'unsigned', 'signed', 'wchar_t', 'auto', 'malloc',
    'free', 'new', 'delete', 'throw', 'try', 'catch', 'finally', 'std', 'deque', 'list', 'queue', 'stack', 'unordered_map',
    'unordered_set', 'pair', 'cmath', 'math', 'pow', 'sqrt', 'abs', 'rand', 'srand',

    # Java Keywords
    'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const', 'continue', 'default', 
    'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements', 'import', 
    'instanceof', 'int', 'interface', 'long', 'native', 'new', 'null', 'package', 'private', 'protected', 'public', 
    'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 
    'try', 'void', 'volatile', 'while', 'String', 'Integer', 'System', 'out', 'println', 'args', 'List', 'ArrayList', 
    'HashMap', 'Set', 'HashSet', 'Runnable', 'Callable', 'Thread', 'Future', 'ExecutorService', 'CompletableFuture', 
    '@Override', '@SuppressWarnings', '@Deprecated'
}"""

common_keywords = {
    # C++ https://en.cppreference.com/w/cpp/keyword
    'alignas', 'alignof', 'and', 'and_eq', 'asm', 'atomic_cancel', 'atomic_commit', 
    'atomic_noexcept', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case', 'catch', 
    'char', 'char8_t', 'char16_t', 'char32_t', 'class', 'compl', 'concept', 'const', 
    'consteval', 'constexpr', 'constinit', 'const_cast', 'continue', 'co_await', 
    'co_return', 'co_yield', 'decltype', 'default', 'delete', 'do', 'double', 
    'dynamic_cast', 'else', 'enum', 'explicit', 'export', 'extern', 'false', 'float', 
    'for', 'friend', 'goto', 'if', 'inline', 'int', 'long', 'mutable', 'namespace', 
    'new', 'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq', 'private', 
    'protected', 'public', 'reflexpr', 'register', 'reinterpret_cast', 'requires', 'return', 
    'short', 'signed', 'sizeof', 'static', 'static_assert', 'static_cast', 'struct', 'switch', 
    'synchronized', 'template', 'this', 'thread_local', 'throw', 'true', 'try', 'typedef', 
    'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile', 
    'wchar_t', 'while', 'xor', 'xor_eq',
    # C https://en.cppreference.com/w/c/keyword
    'alignas', 'alignof', 'auto', 'bool', 'break', 'case', 'char', 'const', 'constexpr', 
    'continue', 'default', 'do', 'double', 'else', 'enum', 'extern', 'false', 'float', 
    'for', 'goto', 'if', 'inline', 'int', 'long', 'nullptr', 'register', 'restrict', 'return', 
    'short', 'signed', 'sizeof', 'static', 'static_assert', 'struct', 'switch', 'thread_local', 
    'true', 'typedef', 'typeof', 'typeof_unqual', 'union', 'unsigned', 'void', 'volatile', 'while',
    '_Alignas', '_Alignof', '_Atomic', '_BitInt', '_Bool', '_Complex', '_Decimal128', '_Decimal32', 
    '_Decimal64', '_Generic', '_Imaginary', '_Noreturn', '_Static_assert', '_Thread_local', 
    'if', 'elif', 'else', 'endif', 'ifdef', 'ifndef', 'elifdef', 'elifndef', 'define', 'undef', 
    'include', 'embed', 'line', 'error', 'warning', 'pragma', 'defined', '__has_include', 
    '__has_embed', '__has_c_attribute',

    # java https://docs.oracle.com/javase/tutorial/java/nutsandbolts/_keywords.html
    'abstract', 'continue', 'for', 'new', 'switch',
    'assert', 'default', 'goto', 'package', 'synchronized',
    'boolean', 'do', 'if', 'private', 'this',
    'break', 'double', 'implements', 'protected', 'throw',
    'byte', 'else', 'import', 'public', 'throws',
    'case', 'enum', 'instanceof', 'return', 'transient',
    'catch', 'extends', 'int', 'short', 'try',
    'char', 'final', 'interface', 'static', 'void',
    'class', 'finally', 'long', 'strictfp', 'volatile',
    'const', 'float', 'native', 'super', 'while','true', 'false','null'
 }

def filterKeywords(content):
    words = re.findall(r'\b\w+\b', content)
    non_generic_words = [word for word in words if word not in common_keywords]
    return non_generic_words

def count_non_generic_terms(content):
        # Split content into words and remove generic terms
        words = re.findall(r'\b\w+\b', content)
        non_generic_words = [word for word in words if word not in common_keywords]

        return len(non_generic_words)

def count_unique_non_generic_terms(content):
        # Split content into words and remove generic terms
        words = re.findall(r'\b\w+\b', content)
        non_generic_words = [word for word in words if word not in common_keywords]
        # Use numpy to get unique words and return their count
        unique_non_generic_words = np.unique(non_generic_words)
        return len(unique_non_generic_words)
"""
def count_comment_lines(code_lines):
    
    #Counts the number of comment lines in the given code_lines for C++, C, and Java code.
    #Includes both single-line (//) and multi-line (/* */) comments.
    

    comment_count = 0
    in_multiline_comment = False

    for line in code_lines:
        stripped_line = line.strip()

        # Check for multi-line comment start
        if in_multiline_comment:
            comment_count += 1  # Count the line inside a multi-line comment
            if "*/" in stripped_line:  # Multi-line comment ends
                in_multiline_comment = False

        elif "//" in stripped_line:
            # Single-line comment
            comment_count += 1

        elif "/*" in stripped_line:
            # Start of a multi-line comment
            comment_count += 1
            in_multiline_comment = True
            if "*/" in stripped_line:  # Multi-line comment ends on the same line
                in_multiline_comment = False

    return comment_count
"""
def count_comment_lines(code_lines):
    """
    Counts the number of comment lines in the given code_lines for C++, C, and Java code.
    Includes both single-line (//) and multi-line (/* */) comments.
    """

    comment_count = 0
    in_multiline_comment = False

    for line in code_lines:
        stripped_line = line.strip()

        # If we're inside a multi-line comment
        if in_multiline_comment:
            comment_count += 1  # Count the line inside a multi-line comment
            if "*/" in stripped_line:  # Multi-line comment ends
                in_multiline_comment = False
            continue  # Skip checking for other comments on this line

        # Check for single-line comments
        if "//" in stripped_line:
            comment_count += 1

        # Check for the start of a multi-line comment
        elif "/*" in stripped_line or "/**" in stripped_line:
            comment_count += 1
            in_multiline_comment = True
            if "*/" in stripped_line:  # Multi-line comment ends on the same line
                in_multiline_comment = False

    return comment_count

def load_code_lines(file_path):
    with open(file_path, 'r') as file:
        code_lines = file.readlines()
    return code_lines

def remove_comments(code_lines): # UNUSED - OK
    code_without_comments = []
    in_multiline_comment = False
    original_line_numbers = []
    for line_num, line in enumerate(code_lines, start=1):
        if not in_multiline_comment:
            if '/*' in line:
                in_multiline_comment = True
                original_line_numbers.append(line_num)
                if '*/' in line:
                    line = line[:line.index('/*')] + line[line.index('*/')+2:]
                else:
                    continue
            elif '//' in line:
                line = line[:line.index('//')]
        if '*/' in line:
            in_multiline_comment = False
            line = line[line.index('*/')+2:]
        if not in_multiline_comment and line.strip():  # Exclude empty lines
            code_without_comments.append(line)
            original_line_numbers.append(line_num)
    return code_without_comments, original_line_numbers

def find_method_points_in_file(file_path,code, output_file_name=True, file_extension='.cpp'):
    method_starts = []
    method_ends = []
    try:
        if file_extension != '.c':
            class_name = find_class_name_in_file(code, file_extension)
            file_name = os.path.basename(file_path).split('.')[0]#.split('_')[0]

            #if use_class_name_to_find_methods:
            #    class_name = find_class_name_in_file(file_path)
                #print('Use Found class name: ', class_name)
            #else:
            #    class_name = os.path.basename(file_path).split('.')[0].split('_')[0]
                #print('Did not use found class name: ', class_name)

            if not class_name:
                return [], []

            #with open(file_path, 'r',errors="ignore") as file:
            #    code = file.readlines()

            if output_file_name:
                output_name = file_name
            else:
                output_name = class_name

            #if use_file_name_to_find_methods:
            #    class_name = file_name

            method_starts = []
            for i, line in enumerate(code):
                if file_extension == '.cpp' or file_extension == '.cc':
                    if class_name.lower() + '::' in line.lower() and line.lower().count(class_name.lower()) == 1:
                        if f"{class_name}::" in line and "(" in line.split(f"{class_name}::")[1]:
                            # Initialize bracket flags
                            open_bracket_found = False
                            closing_bracket_found = False

                            # Now check if '{' appears, but not '}' before it
                            j = i
                            while j < len(code):
                                current_line = code[j].strip()

                                # Check for '}' before '{'
                                if '}' in current_line:
                                    closing_bracket_found = True
                                if '{' in current_line:
                                    open_bracket_found = True

                                # If we found '}', stop looking and discard this method
                                if closing_bracket_found and not open_bracket_found:
                                    break

                                # If we found '{', confirm the method and stop
                                if open_bracket_found:
                                    method_starts.append((output_name, i))  # Method starts at 'i'
                                    break

                                # Move to the next line
                                j += 1

                elif file_extension == '.java':
                    #method_pattern = re.compile(r'^\s*(public|protected|private|static|final|abstract|synchronized|native|void)\s+\S+\s+\w+\s*\(.*?\)\s*\{', re.DOTALL | re.MULTILINE)
                    method_pattern = re.compile(
                        r'^\s*(public|protected|private|static|final|abstract|synchronized|native|void)\s+\S+\s+\w+\s*\(.*?\)\s*(throws\s+\w+(\s*,\s*\w+)*)?\s*\{',
                        re.DOTALL | re.MULTILINE
                    )
                    """
                    method_pattern = re.compile(
                        r'''
                        ^\s*                                            # Leading whitespace
                        (?:/\*\*.*?\*/\s*)?                             # Optional JavaDoc comment (non-greedy)
                        (?:@[^\s]+\s*)*                                 # Optional annotations (e.g., @Override)
                        (public|protected|private|static|final|abstract|synchronized|native)?\s*  # Optional method modifiers
                        (?:\w+<.*?>\s+)?                                # Optional generic return type
                        \w+\s+                                          # Return type or `void`
                        \w+\s*                                          # Method name
                        \(\s*.*?\s*\)                                   # Parameters (allow multiline, non-greedy)
                        (\s*throws\s+\w+(\s*,\s*\w+)*)?\s*              # Optional throws clause
                        \{                                              # Opening brace
                        ''',
                        re.DOTALL | re.MULTILINE | re.VERBOSE
                    )"""

                    if method_pattern.match(line):
                        #print('Found method: ', output_name, i)
                        method_starts.append((output_name, i))
                        """
                    if class_name.lower() + '::' in line.lower() and line.lower().count(class_name.lower()) == 1:
                        if f"{class_name}::" in line and "(" in line.split(f"{class_name}::")[1]:
                            # Initialize bracket flags
                            open_bracket_found = False
                            closing_bracket_found = False

                            # Now check if '{' appears, but not '}' before it
                            j = i
                            while j < len(code):
                                current_line = code[j].strip()

                                # Check for '}' before '{'
                                if '}' in current_line:
                                    closing_bracket_found = True
                                if '{' in current_line:
                                    open_bracket_found = True

                                # If we found '}', stop looking and discard this method
                                if closing_bracket_found and not open_bracket_found:
                                    break

                                # If we found '{', confirm the method and stop
                                if open_bracket_found:
                                    method_starts.append((output_name, i))  # Method starts at 'i'
                                    break
                                # Move to the next line
                                j += 1"""


            method_ends = [] 
            for start_info in method_starts:
                class_name, start = start_info
                brace_count = 0
                opening_brace_found = False  # Ensure the opening brace is found
                for i in range(start, len(code)):
                    line = code[i]
                    if '{' in line:  # Mark the opening brace as found
                        opening_brace_found = True
                    if opening_brace_found:  # Only count braces after the opening brace is found
                        brace_count += line.count('{') - line.count('}')
                        if brace_count == 0 and i != start:
                            method_ends.append(i)
                            break

    except UnicodeDecodeError:
        pass
    return method_starts, method_ends

def find_class_name_in_file(code, file_extension='.cpp'):
    class_name = None
    # Ensure `code` is a string; if it's a list, join lines
    if isinstance(code, list):
        code = "\n".join(code)
    #with open(file_path, 'r') as file:
    #    code = file.read()

    if file_extension == '.cpp' or file_extension == '.cc':
        # Find the pattern '[classname]::[classname]' (case-insensitive) at the start of each line
        class_match = re.search(r'^\s*(\w+)::\1\b', code, re.MULTILINE | re.IGNORECASE)

        if class_match:
            class_name = class_match.group(1)
        else:
            # Find the pattern '[classname]::~[classname]' (case-insensitive) at the start of each line
            destructor_match = re.search(r'^\s*(\w+)::~\1\b', code, re.MULTILINE | re.IGNORECASE)

            if destructor_match:
                class_name = destructor_match.group(1)
            else:
                # Find the pattern 'bool [classname]::' (case-insensitive) at the start of each line
                bool_match = re.search(r'^\s*bool\s+(\w+)::\b', code, re.MULTILINE | re.IGNORECASE)

                if bool_match:
                    class_name = bool_match.group(1)
                else:
                    # Find the pattern 'void [classname]::' (case-insensitive) at the start of each line
                    pattern_match = re.search(r'^\s*void\s+(\w+)::', code, re.MULTILINE | re.IGNORECASE)

                    if pattern_match:
                        class_name = pattern_match.group(1)
    elif file_extension == '.java':
        class_match = re.search(r'((public|private|protected|final|abstract|static|@interface)\s+class)\s+(\w+)', code, re.IGNORECASE)
        if class_match:
            class_name = class_match.group(3)
            #print('Found class name: ', class_name)

    return class_name

def find_functions_in_c_file(code):
    #with open(file_path, 'r', errors="ignore") as file:
    #    code = file.readlines()

    function_starts = []
    function_ends = []

    # Initialize boolean array to track where function declarations may start
    can_start_function = [True] * len(code)

    # Precompile patterns to check for lines that cannot start a function declaration
    invalid_start_patterns = [
        r'^\s',  # Lines starting with whitespace
        r'^\t', #tab
        r'^\/\*',  # Lines starting with block comments
        r'^\/\/',  # Lines starting with single-line comments
        r'^#',  # Preprocessor directives
        r'^return',  # Return statement (not a function declaration)
        r'^\*',  # Continuation of a multi-line comment
        r'^@',  # Special symbols, e.g., used for decorators or annotations
        r'^\}',  # Closing bracket on its own line
        r'.*\)\s*;',  # Macro-like declarations ending with );
        r'^&&',
        r'for'
    ]

    # Mark lines that cannot start a function declaration
    for i, line in enumerate(code):
        for pattern in invalid_start_patterns:
            if re.match(pattern, line.strip()):  # Strip leading/trailing whitespaces for better matching
                can_start_function[i] = False
                break

    # Regex patterns for detecting function definitions in C
    function_start_pattern = re.compile(r'^[^\s]+[\w\*\s]+\w+\s*\(.*\)$')  # Single line function signature with no leading spaces
    multiline_start_pattern = re.compile(r'^[^\s]+[\w\*\s]+\w+\s*\([^)]*$')  # Multi-line function signature start
    signature_end_pattern = re.compile(r'^\s*\)\s*{$')  # End of signature with opening brace

    brace_count = 0
    inside_function = False
    function_start = None

    i = 0
    while i < len(code):
        line = code[i].strip()  # Strip any leading/trailing whitespace for better matching

        # Skip lines that cannot start a function
        if not can_start_function[i]:
            i += 1
            continue

        # Detect function start (single-line or multiline start)
        if function_start_pattern.match(line) or multiline_start_pattern.match(line):
            if inside_function:
                # End the previous function (in case of missing closure)
                function_ends.append(i - 1)
                inside_function = False

            # Start of a new function
            function_start = i
            function_starts.append(function_start)
            inside_function = True

            # Check for opening brace on the same line (single-line function header)
            if '{' in line:
                brace_count += line.count('{')

            i += 1
            continue

        # Handle multi-line function declarations and detect the end of the function signature
        if inside_function and signature_end_pattern.match(line):
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0:
                function_ends.append(i)  # Close the function
                inside_function = False

        i += 1

    # Handle any unclosed function at the end of the file
    if inside_function:
        function_ends.append(len(code) - 1)

    # Ensure function_ends list matches the length of function_starts list
    if len(function_starts) > len(function_ends):
        function_ends.extend([len(code) - 1] * (len(function_starts) - len(function_ends)))

    return function_starts, function_ends

def remove_class_name_from_method_content(method_content, class_name):
    # Find the index of class name in the method content
    index = method_content.lower().find(class_name.lower())
    if index != -1:
        # Remove the class name from the method content along with the following '::' if present TODO remove even with no '::'
        end_index = method_content.find('::', index + len(class_name))
        if end_index != -1:
            method_content = method_content[:index] + method_content[end_index + 2:]
        else:
            method_content = method_content[:index] + method_content[index + len(class_name):]
    return method_content

def get_include_lines(code_lines, system_name, file_name, directory_path):
    """
    Returns a list of lines where external include directives are found and the first line
    where the file name is mentioned after splitting by '\'.
    Only considers include_string occurrences after '#include' or the relevant keyword.
    """

    import os

    # Extract the part of the file name after splitting by '\'
    file_name_part = file_name.split("\\")[-1]

    split_path = directory_path.split('code')
    remaining_path = split_path[1]
    # Split the remaining path by os.sep
    class_folder = remaining_path.strip(os.sep).split(os.sep)[0]

    include_lines = []
    first_file_include_line = None

    for line in code_lines:
        # Strip leading/trailing whitespaces
        stripped_line = line.strip()

        # Check for system-specific include lines
        if system_name == 'autoware':
            include_string = '#include'  # '#include <autoware'
            if include_string in stripped_line and 'hpp' in stripped_line and class_folder not in stripped_line and file_name_part not in stripped_line:
                include_lines.append(line)

        elif system_name == 'teammates':
            include_string = 'import teammates.'
            if include_string in stripped_line and class_folder not in stripped_line and file_name_part not in stripped_line:
                include_lines.append(line)

        elif system_name == 'opencv':
            include_string = '#include "opencv2'
            include_string2 = '#include <opencv2'
            if (include_string in stripped_line or include_string2 in stripped_line) and file_name_part not in stripped_line:
                include_lines.append(line)

        elif system_name == 'rtems':
            include_string = '#include <rtems'
            if include_string in stripped_line and class_folder not in stripped_line and file_name_part not in stripped_line:
                include_lines.append(line)

        # Check if the file name part is present in the line
        if first_file_include_line is None and file_name_part in stripped_line:
            first_file_include_line = line

    return include_lines, first_file_include_line



def get_number_of_ext_includes(code_lines, system_name, file_name, directory_path):
    """
    Counts the number of occurrences of external include directives in the file or method.
    Only counts include_string occurrences after '#include' or the relevant keyword.
    """

    split_path = directory_path.split('code')
    # Extract the part of the file name after splitting by '\'
    file_name_part = file_name.split("\\")[-1]
    remaining_path = split_path[1]
    # Split the remaining path by os.sep
    class_folder = remaining_path.strip(os.sep).split(os.sep)[0]

    ext_include_count = 0
    
    for line in code_lines:
        # Strip leading/trailing whitespaces
        stripped_line = line.strip()

        if system_name == 'autoware':
            include_string = '#include'#'#include <autoware'
            if include_string in stripped_line:# and 'hpp' in stripped_line and class_folder not in stripped_line and file_name_part not in stripped_line:
                ext_include_count += 1

        elif system_name == 'teammates':
            include_string = 'import'# teammates.'
            if include_string in stripped_line:# and class_folder not in stripped_line and file_name_part not in stripped_line:
                ext_include_count += 1

        elif system_name == 'opencv':
            include_string = '#include'# "opencv2'
            include_string2 = '#include'# <opencv2'
            if include_string in stripped_line or include_string2 in stripped_line: 
                if file_name_part not in stripped_line:
                    ext_include_count += 1

        elif system_name == 'rtems':
            include_string = '#include'# <rtems'
            if include_string in stripped_line: #and class_folder not in stripped_line and file_name_part not in stripped_line:
                ext_include_count += 1
    
    return ext_include_count

def extract_methods_from_file(file_path,root_directory, system_name, output_file_name=True, file_extension='.cpp'):
    methods_data = []
    if file_extension != '.c':
        
        with open(file_path, 'r', errors="ignore") as file:
            code = file.readlines()
            file_name = os.path.basename(file_path).split('.')[0]
            n_of_ext_includes = get_number_of_ext_includes(code, system_name, file_name, file_path)
            include_lines, self_include_line = get_include_lines(code, system_name, file_name, file_path)
            code = remove_initial_comment_block(code)
            method_starts, method_ends = find_method_points_in_file(file_path,code, output_file_name, file_extension)

            
        # Convert file path to relative path
        relative_file_path = os.path.relpath(file_path, root_directory)

        for start_info, end in zip(method_starts, method_ends):
            class_name, start = start_info
            method_lines = code[start:end + 1]
            method_content = "".join(method_lines)
            n_comments = count_comment_lines(method_lines)
            #method_content = remove_class_name_from_method_content(method_content, class_name)
            
            if len(method_content.split("(")[0].strip().split()) > 0:
                method_name = method_content.split("(")[0].strip().split()[-1]
                methods_data.append({
                    'Class': relative_file_path, 
                    'Method Name': method_name, 
                    'Start Line': start + 1,
                    'End Line': end + 1, 
                    'Content': method_content,
                    'n_non_generic_terms': count_non_generic_terms(method_content),
                    'n_unique_non_generic_terms': count_unique_non_generic_terms(method_content),
                    'n_ext_includes': n_of_ext_includes,
                    'include lines': include_lines,
                    'n_comments': n_comments
                    #'self include line': self_include_line
                })
    elif file_extension == '.c':
        
        file_name = os.path.basename(file_path).split('.')[0]
        with open(file_path, 'r', errors="ignore") as file:
            code = file.readlines()
            n_of_ext_includes = get_number_of_ext_includes(code, system_name, file_name, file_path)
            include_lines, self_include_line = get_include_lines(code, system_name, file_name, file_path)
            code = remove_initial_comment_block(code)
            method_starts, method_ends = find_functions_in_c_file(code)
        for start in method_starts:
            try:
                start_line = start
                end_line = method_ends[method_starts.index(start)]

                method_lines = code[start_line:end_line + 1]
                method_content = "".join(method_lines)
                n_comments = count_comment_lines(method_lines)

                #method_content = "".join(code[start_line:end_line + 1])
                #n_comments = count_comment_lines(method_content)
                relative_file_path = os.path.relpath(file_path, root_directory)
                if '{' in method_content and '}' in method_content and not method_content.startswith((' ', '\t')):
                    method_name_match = re.search(r'(\w+)\s*\(', method_content)
                    method_name = method_name_match.group(1) if method_name_match else "Unknown"
                    methods_data.append({
                        'Class': relative_file_path,
                        'Method Name': method_name,
                        'Start Line': start_line + 1,
                        'End Line': end_line + 1,
                        'Content': method_content,
                        'n_non_generic_terms': count_non_generic_terms(method_content),
                        'n_unique_non_generic_terms': count_unique_non_generic_terms(method_content),
                        'n_ext_includes': n_of_ext_includes,
                        'include lines': include_lines,
                        'n_comments': n_comments
                        #'self include line': self_include_line
                    })
            except Exception as e:
                print(f"Error processing method at line {start}: {e}")
    return methods_data

def check_comment_line(line):
    """
    Check if the line is a comment line (single-line comment).
    Handles both '//' and ' //', and specific comment markers like '//M' and ' //M'.
    """
    return line.strip().startswith("//")

def check_comment_block_start(line):
    """
    Check if the line starts a block comment.
    Handles both '/*' and ' /*'.
    """
    return "/*" in line.strip() or "/**" in line.strip()

def check_comment_block_end(line):
    """
    Check if the line ends a block comment.
    Handles both '*/' and '*/M'.
    """
    return "*/" in line.strip()

def remove_initial_comment_block(file_content):
    """
    Removes the initial comment block from the extracted file content, including multi-line comment blocks
    and blocks of consecutive single-line comments starting with '//' or similar patterns.
    
    Parameters:
    - file_content: A list of strings, where each string is a line from the code file.
    
    Returns:
    - A list of strings representing the file content with the initial comment block removed.
    """
    in_comment_block = False
    result = []
    save_following_lines = False
    skip_single_line_block = False

    for i, line in enumerate(file_content):
        stripped_line = line.strip()

        if save_following_lines:
            # Once we've decided to save lines, keep appending them
            result.append(line)
            continue

        # Detect the start of multi-line comment
        if not in_comment_block:
            if check_comment_block_start(line):
                in_comment_block = True
                continue  # Skip this line

        # Handle the end of a multi-line comment block
        if in_comment_block:
            if check_comment_block_end(line):
                in_comment_block = False  # End the block comment
                continue  # Skip the ending line
            continue  # Skip lines inside the comment block

        # Detect blocks of consecutive single-line comments
        if check_comment_line(line):
            if i + 1 < len(file_content) and check_comment_line(file_content[i + 1]):
                skip_single_line_block = True
                continue
            elif skip_single_line_block:
                if not check_comment_line(file_content[i - 1]):  # End of the consecutive block
                    skip_single_line_block = False
                continue
            else:
                result.append(line)  # Save standalone single-line comments
                continue

        # Stop skipping when we find a line with text outside the comment block
        if stripped_line:
            save_following_lines = True
            result.append(line)

    return result



# #     return result
# def remove_initial_comment_block(file_content):
#     """
#     Removes all initial comment blocks (multi-line and single-line) from the beginning of the file content.

#     Parameters:
#     - file_content: A list of strings, where each string is a line from the code file.

#     Returns:
#     - A list of strings representing the file content with initial comment blocks removed.
#     """
#     result = []
#     in_comment_block = False

#     for line in file_content:
#         stripped_line = line.strip()

#         # Handle start of multi-line comment block
#         if "/*" in stripped_line and not in_comment_block:
#             in_comment_block = True
#             continue

#         # Handle end of multi-line comment block
#         if in_comment_block:
#             if "*/" in stripped_line:
#                 in_comment_block = False
#             continue

#         # Handle single-line comments
#         if stripped_line.startswith("//"):
#             continue

#         # Stop skipping when we reach the first non-comment line
#         if not in_comment_block and not stripped_line.startswith("/*"):
#             result.append(line)
#             break

#     # Add remaining lines after the first non-comment block
#     result.extend(file_content[len(result):])

#     return result



def extract_whole_file_content(file_path, root_directory, system_name):
    """Extract the entire file content as a single record."""
    try:
        with open(file_path, 'r', errors="ignore") as file:
            lines = file.readlines()
            file_name = os.path.basename(file_path).split('.')[0]
            n_of_ext_includes = get_number_of_ext_includes(lines, system_name, file_name, file_path)
            include_lines = get_include_lines(lines, system_name, file_name, file_path)

            # Remove the initial comment block
            lines = remove_initial_comment_block(lines)
            n_comments = count_comment_lines(lines)
            # Join lines to form the entire content as one string
            file_content = ''.join(lines)
        # Convert file path to relative path
        relative_file_path = os.path.relpath(file_path, root_directory)
        return [{
            'Class': relative_file_path,
            'Content': file_content,
            'Lines of code': len(lines),  # Number of lines in the file
            'n_non_generic_terms': count_non_generic_terms(file_content), # 'n_non_generic_terms': count_non_generic_terms(method_content),
            'n_unique_non_generic_terms': count_unique_non_generic_terms(file_content),
            'n_ext_includes': n_of_ext_includes,
            'n_comments': n_comments
            #'include lines': include_lines
        }]
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    

def extract_methods_identifiers_from_directory(directory_path, system_name, file_extension, extract_whole_file=False, output_file_name=True):
    all_method_data = []

    for root, _, files in os.walk(directory_path):
        # Skip any directory in the path that contains parentheses
        if '(' in root or ')' in root:
            continue

        for file in files:
            if 'test' not in file and file.endswith(file_extension):
                file_path = os.path.join(root, file)
                
                if extract_whole_file:
                    method_data = extract_whole_file_content(file_path, directory_path, system_name)
                else:
                    method_data = extract_methods_from_file(file_path, directory_path, system_name, output_file_name, file_extension)
                
                all_method_data.extend(method_data)

    # Then pass it to pd.DataFrame
    df = pd.DataFrame(all_method_data)
    df = pd.DataFrame(all_method_data).fillna('')
    if not extract_whole_file:
        df['Lines of code'] = df['End Line'] - df['Start Line']

    df = df[df['Lines of code'] > 1] # remove incorrect findings

    df['Id'] = range(0, len(df))
    
    return df 

def merge_methods_from_same_file(df):
    # Group by 'Class Name' to find all methods from the same file/class
    merged_df = df.groupby('Class').agg({
        'Content': lambda x: "\n".join(x),  # Concatenate all method contents
        'Lines of code': 'sum',  # Sum the method lengths
        'Start Line': 'min',  # Keep the minimum start line as the start of the file
        'End Line': 'max'  # Keep the maximum end line as the end of the file
    }).reset_index()

    merged_df['Lines of code'] = merged_df['End Line'] - merged_df['Start Line']  # Calculate total length of the file
    merged_df['Id'] = range(0, len(merged_df))
    return merged_df