import glob
import fnmatch

# Define the ignore list with glob patterns you want to exclude.
ignore_list = ["**/__init__.py", "**/question_builder.py", "**/venv/**"]


# Function to check if the file matches any pattern in the ignore list.
def is_ignored(file_path, ignore_patterns):  # Avoiding shadowing 'ignore_list'
    for pattern in ignore_patterns:
        if fnmatch.fnmatch(file_path, pattern):  # Using imported fnmatch
            return True
    return False


# Search for `.py` files and exclude those in the ignore list.
def find_py_files(base_dir, ignore_patterns):  # Renamed parameter to avoid shadowing
    all_py_files = glob.glob(f"{base_dir}/**/*.py", recursive=True)
    return [file for file in all_py_files if not is_ignored(file, ignore_patterns)]


# Read the content of each `.py` file.
def read_file_content(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# Main function to generate the markdown file.
def generate_markdown(
    base_dir, ignore_patterns, output_file
):  # Adjust to the new parameter name
    py_files = find_py_files(base_dir, ignore_patterns)
    with open(output_file, "w", encoding="utf-8") as md_file:
        for py_file in py_files:
            content = read_file_content(py_file)
            md_file.write(f"{py_file}\n```python\n{content}\n```\n\n")


if __name__ == "__main__":
    # Example usage:
    base_directory = "."  # Current directory as base. Change as needed.
    output_markdown_file = "codebase.txt"  # Output markdown file name.
    generate_markdown(base_directory, ignore_list, output_markdown_file)
