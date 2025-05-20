#!/usr/bin/env python3

# This script updates the sample size extraction in parallel_rct.py
import re

file_path = "./app/components/parallel_rct.py"

with open(file_path, 'r') as f:
    content = f.read()

# Find the binary sample size extraction code and update it
pattern = r"        # Extract values from result\n        n1 = sample_size\.get\(\"n1\", 0\)\n        n2 = sample_size\.get\(\"n2\", 0\)\n        total_n = sample_size\.get\(\"total_n\", n1 \+ n2\)"

replacement = """        # Extract values from result - handle different key names from different functions
        # The analytical function returns sample_size_1, sample_size_2, total_sample_size
        n1 = sample_size.get("sample_size_1", sample_size.get("n1", 0))
        n2 = sample_size.get("sample_size_2", sample_size.get("n2", 0))
        total_n = sample_size.get("total_sample_size", sample_size.get("total_n", n1 + n2))"""

updated_content = re.sub(pattern, replacement, content)

with open(file_path, 'w') as f:
    f.write(updated_content)

print("Updated sample size extraction code in parallel_rct.py")
