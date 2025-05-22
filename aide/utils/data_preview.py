# aide/utils/data_preview.py

import json
from pathlib import Path
import logging # Import logging

import humanize
import pandas as pd
from genson import SchemaBuilder
from pandas.api.types import is_numeric_dtype

logger = logging.getLogger("aide.data_preview") # Specific logger for this module

# these files are treated as code (e.g. markdown wrapped)
code_files = {".py", ".sh", ".yaml", ".yml", ".md", ".html", ".xml", ".log", ".rst"}
# we treat these files as text (rather than binary) files
plaintext_files = {".txt", ".csv", ".json", ".tsv"} | code_files


def get_file_len_size(f: Path) -> tuple[int, str]:
    """
    Calculate the size of a file (#lines for plaintext files, otherwise #bytes)
    Also returns a human-readable string representation of the size.
    """
    try:
        if f.suffix in plaintext_files:
            # Ensure file is read with an encoding that handles potential diverse characters
            with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
                num_lines = sum(1 for _ in fp)
            return num_lines, f"{num_lines} lines"
        else:
            s = f.stat().st_size
            return s, humanize.naturalsize(s)
    except Exception as e:
        logger.error(f"Error getting size for file {f}: {e}")
        return 0, "Error reading file"


def file_tree(path: Path, depth=0) -> str:
    """Generate a tree structure of files in a directory"""
    result = []
    try:
        # Separate files and directories, handling potential permission errors
        items = list(Path(path).iterdir())
        files = sorted([p for p in items if p.is_file()])
        dirs = sorted([p for p in items if p.is_dir()])
    except PermissionError:
        logger.warning(f"Permission denied reading directory: {path}")
        return f"{' '*depth*4} [Permission Denied]"
    except Exception as e:
        logger.error(f"Error iterating directory {path}: {e}")
        return f"{' '*depth*4} [Error Reading Directory]"


    max_n_files = 4 if len(files) > 30 else 8
    for p in files[:max_n_files]:
        result.append(f"{' '*depth*4}{p.name} ({get_file_len_size(p)[1]})")
    if len(files) > max_n_files:
        result.append(f"{' '*depth*4}... and {len(files)-max_n_files} other files")

    max_n_dirs = 10 # Limit number of directories listed to prevent excessively long trees
    for p in dirs[:max_n_dirs]:
        result.append(f"{' '*depth*4}{p.name}/")
        result.append(file_tree(p, depth + 1))
    if len(dirs) > max_n_dirs:
        result.append(f"{' '*depth*4}... and {len(dirs)-max_n_dirs} other directories")


    return "\n".join(result)


def _walk(path: Path):
    """Recursively walk a directory (analogous to os.walk but for pathlib.Path)"""
    try:
        for p in sorted(Path(path).iterdir()):
            if p.is_dir():
                yield from _walk(p)
            else: # It's a file
                yield p
    except PermissionError:
        logger.warning(f"Permission denied walking directory: {path}")
    except Exception as e:
        logger.error(f"Error walking directory {path}: {e}")


def preview_csv(p: Path, file_name: str, simple=True) -> str:
    try:
        # Try to read with common encodings, and handle potential parsing errors
        try:
            df = pd.read_csv(p, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding='latin1')
        except pd.errors.ParserError as pe:
            logger.warning(f"Pandas parsing error for CSV {file_name}: {pe}. Skipping detailed preview.")
            return f"-> {file_name} could not be parsed as a standard CSV. File size: {get_file_len_size(p)[1]}."
        except Exception as e: # Catch other pandas read_csv errors
            logger.error(f"Error reading CSV {file_name}: {e}")
            return f"-> Error reading CSV {file_name}: {e}"


        out = [f"-> {file_name} has {df.shape[0]} rows and {df.shape[1]} columns."]

        if simple:
            cols = df.columns.tolist()
            sel_cols = 15
            cols_str = ", ".join(cols[:sel_cols])
            res = f"The columns are: {cols_str}"
            if len(cols) > sel_cols:
                res += f"... and {len(cols)-sel_cols} more columns"
            out.append(res)
        else:
            out.append("Here is some information about the columns:")
            for col_idx, col_name_original in enumerate(df.columns):
                # Sanitize column name for processing if it's not a string (e.g. int from bad header)
                col = str(col_name_original)
                
                # Check if the original column name (if different after str conversion) exists
                # This handles cases where pandas might infer integer column names
                actual_col_data = df.iloc[:, col_idx] if col not in df else df[col]

                dtype = actual_col_data.dtype
                name_display = f"{col} ({dtype})" # Use sanitized name for display

                nan_count = actual_col_data.isnull().sum()

                if dtype == "bool":
                    v = actual_col_data[actual_col_data.notnull()].mean()
                    out.append(f"{name_display} is {v*100:.2f}% True, {100-v*100:.2f}% False")
                elif actual_col_data.nunique() < 10:
                    unique_vals = actual_col_data.unique().tolist()
                    # Truncate long lists of unique values
                    if len(unique_vals) > 5: unique_vals_display = str(unique_vals[:5]) + "..."
                    else: unique_vals_display = str(unique_vals)
                    out.append(f"{name_display} has {actual_col_data.nunique()} unique values: {unique_vals_display}")
                elif is_numeric_dtype(actual_col_data):
                    out.append(f"{name_display} has range: {actual_col_data.min():.2f} - {actual_col_data.max():.2f}, {nan_count} nan values")
                elif dtype == "object":
                    # Example values: take top 4, ensure they are strings for display
                    example_values = [str(val) for val in actual_col_data.value_counts().head(4).index.tolist()]
                    out.append(f"{name_display} has {actual_col_data.nunique()} unique values. Some example values: {example_values}")
        return "\n".join(out)
    except Exception as e:
        logger.error(f"Failed to preview CSV {file_name}: {e}")
        return f"-> Error previewing CSV {file_name}: {e}. File size: {get_file_len_size(p)[1]}."


def preview_json(p: Path, file_name: str) -> str:
    """Generate a textual preview of a json file using a generated json schema"""
    builder = SchemaBuilder()
    try:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            # Attempt to read the first line to check for JSONL
            try:
                first_line = f.readline().strip()
                if not first_line: # Empty file
                    return f"-> {file_name} is empty."
                
                first_object = json.loads(first_line)

                # Check if it's likely JSONL
                # A simple heuristic: if the first line is a valid JSON object
                # and there's more content in the file, assume JSONL.
                # This isn't foolproof but better than trying to read the whole file if it's huge.
                # We also need to ensure first_object is a dict or list for genson.
                if isinstance(first_object, (dict, list)):
                    # Try to read the next line
                    try:
                        second_line = f.readline().strip() # Read up to a certain number of bytes for performance
                    except Exception: # Could be very large line
                        second_line = "" # Assume not JSONL or handle as error

                    if second_line: # If there's a second line, assume JSONL
                        f.seek(0) # Reset to read all lines
                        # Limit the number of lines processed for schema generation to avoid OOM on huge JSONL
                        lines_processed = 0
                        max_lines_for_schema = 100 # Configurable limit
                        for line_content in f:
                            if lines_processed >= max_lines_for_schema:
                                logger.info(f"Reached max_lines_for_schema ({max_lines_for_schema}) for {file_name}. Schema might be partial.")
                                break
                            try:
                                builder.add_object(json.loads(line_content.strip()))
                                lines_processed +=1
                            except json.JSONDecodeError as line_e:
                                logger.warning(f"Skipping invalid JSON line in {file_name}: {line_e} - Content: {line_content[:100]}...")
                                continue # Skip malformed lines in JSONL
                        schema_info = " (schema from first " + str(lines_processed) + " lines)" if lines_processed >= 1 else ""

                    else: # Only one line, and it was a valid JSON object
                        builder.add_object(first_object)
                        schema_info = " (schema from single JSON object)"
                else: # First line was valid JSON, but not a dict/list (e.g. just a string or number)
                    f.seek(0) # Read the whole file as one JSON entity
                    builder.add_object(json.load(f)) # This might fail if it's not valid JSON overall
                    schema_info = " (schema from full file content)"

            except json.JSONDecodeError: # First line wasn't valid JSON, assume multi-line single JSON object
                f.seek(0)
                builder.add_object(json.load(f)) # This is where your original error happened
                schema_info = " (schema from full file content, assuming single multi-line JSON)"
            except Exception as e_read: # Catch other read errors
                logger.error(f"Error reading lines from JSON file {file_name} for schema generation: {e_read}")
                return f"-> Error reading JSON file {file_name} for schema: {e_read}"

        return f"-> {file_name} has auto-generated json schema{schema_info}:\n" + builder.to_json(indent=2)

    except json.JSONDecodeError as e: # This catches the error from the final json.load(f) if it fails
        logger.error(f"Failed to parse {file_name} as JSON: {e}. Error at char {e.pos}.")
        return f"-> Error: {file_name} is not a valid JSON file or contains syntax errors (e.g., unterminated string at char {e.pos}). File size: {get_file_len_size(p)[1]}."
    except Exception as e:
        logger.error(f"Unexpected error previewing JSON {file_name}: {e}")
        return f"-> Unexpected error previewing JSON {file_name}: {e}. File size: {get_file_len_size(p)[1]}."


def generate(base_path: Path, include_file_details=True, simple=False) -> str: # Added Path type hint
    """
    Generate a textual preview of a directory, including an overview of the directory
    structure and previews of individual files
    """
    if not isinstance(base_path, Path): # Ensure base_path is a Path object
        base_path = Path(base_path)

    logger.info(f"Generating data preview for: {base_path}")
    tree_str = file_tree(base_path)
    tree = f"Directory structure for {base_path.name}:\n```\n{tree_str}\n```"
    out = [tree]

    if include_file_details:
        files_processed_count = 0
        max_files_to_detail = 10 # Limit number of files to detail to prevent overly long previews

        for fn in _walk(base_path): # _walk should also be robust
            if files_processed_count >= max_files_to_detail:
                logger.info(f"Reached max_files_to_detail ({max_files_to_detail}). Skipping details for remaining files.")
                out.append(f"... and more files (details omitted for brevity).")
                break
            
            file_name = str(fn.relative_to(base_path))
            preview_text = None

            try:
                if fn.suffix == ".csv":
                    preview_text = preview_csv(fn, file_name, simple=simple)
                elif fn.suffix == ".json":
                    preview_text = preview_json(fn, file_name)
                elif fn.suffix in plaintext_files:
                    # For plaintext, only show content if small, otherwise just mention existence and size
                    file_len, file_size_str = get_file_len_size(fn)
                    if file_len < 30 and file_len > 0 : # Show content for small files
                        with open(fn, 'r', encoding='utf-8', errors='ignore') as f_content:
                            content = f_content.read()
                        if fn.suffix in code_files:
                            content = f"```\n{content}\n```"
                        preview_text = f"-> {file_name} ({file_size_str}) has content:\n\n{content}"
                    else: # For larger plaintext files, just note their existence
                        preview_text = f"-> {file_name} ({file_size_str}) exists."
                # else: skip binary files or add specific handlers

                if preview_text:
                    out.append(preview_text)
                    files_processed_count += 1

            except Exception as e_file_prev:
                logger.error(f"Error generating preview for file {file_name}: {e_file_prev}")
                out.append(f"-> Error generating preview for {file_name}: {e_file_prev}")
                files_processed_count += 1 # Count it as processed even if error


    result = "\n\n".join(out)
    max_len = 10000 # Increased max length a bit, but still needs a limit

    if len(result) > max_len:
        if not simple: # If it's too long and not already simple, try simple
            logger.info(f"Data preview length ({len(result)}) > max_len ({max_len}). Retrying with simple=True.")
            return generate(base_path, include_file_details=include_file_details, simple=True)
        else: # If already simple and still too long, truncate
            logger.info(f"Data preview length ({len(result)}) > max_len ({max_len}) even with simple=True. Truncating.")
            return result[:max_len] + f"\n... (data preview truncated at {max_len} characters)"

    return result
