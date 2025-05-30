import os

OUTPUT_MD = "FULL_CODEBASE.md"

# Directories to scan
DIRS = [
    "aide",
    "aide/backend",
    "aide/utils",
    "."
]

# Helper to get all .py files in a directory (non-recursive for backend/utils, recursive for aide)
def get_py_files(base_dir):
    py_files = []
    for root, dirs, files in os.walk(base_dir):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        dirs[:] = [d for d in dirs if d != ".venv"]
        dirs[:] = [d for d in dirs if d != "data"]
        dirs[:] = [d for d in dirs if d != "helpers"]
        for f in files:
            if f.endswith(".py") and not f.startswith("."):
                rel_path = os.path.relpath(os.path.join(root, f), ".")
                py_files.append(rel_path)
        # For backend/utils, don't recurse
        if base_dir in ["aide/backend", "aide/utils"]:
            break
    return py_files

all_py_files = set()
for d in DIRS:
    if os.path.isdir(d):
        all_py_files.update(get_py_files(d))

# Remove duplicates and sort
all_py_files = sorted(all_py_files)
failed_files = []
with open(OUTPUT_MD, "w") as out:
    out.write("# Full aide-agent Codebase\n\n")
    for path in all_py_files:
        out.write(f"## {path}\n\n")
        out.write("```python\n")
        try:
            with open(path, "r") as f:
                out.write(f.read())
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            failed_files.append(path)
        out.write("\n```")
        out.write("\n\n")

print(f"Wrote {len(all_py_files)} files to {OUTPUT_MD}") 
print(f"Failed to read {len(failed_files)} files")