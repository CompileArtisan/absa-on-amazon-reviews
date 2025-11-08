from pathlib import Path

output = Path("combined_output.txt")
output.write_text("")  # clear file

for file in sorted(Path(".").glob("*_OUTPUT.txt")):
    if file.is_file():
        with output.open("a") as out:
            out.write(f"===== {file.name} =====\n")
            out.write(file.read_text())
            out.write("\n\n")

print("Combined output saved to combined_output.txt")
