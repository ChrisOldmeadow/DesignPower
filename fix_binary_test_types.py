#!/usr/bin/env python3

# This script updates the test type mappings in parallel_rct.py
import re

file_path = "./app/components/parallel_rct.py"

with open(file_path, 'r') as f:
    content = f.read()

# Fix the power calculation section
power_pattern = r"""            # Convert test name to test_type expected by the function
            test_name_mapping = \{
                "chi-square": "normal approximation",
                "fisher's exact": "fishers exact",
                "z-test": "normal approximation"
            \}
            mapped_test_type = test_name_mapping\.get\(test\.lower\(\), "normal approximation"\)"""

power_replacement = """            # Convert test name to test_type expected by the function
            test_name_mapping = {
                "normal approximation": "normal approximation",
                "fisher's exact test": "fishers exact",
                "likelihood ratio test": "likelihood ratio"
            }
            # Convert the test_type to lowercase for case-insensitive matching
            mapped_test_type = test_name_mapping.get(test_type.lower(), "normal approximation")"""

# Fix the MDE calculation section 
mde_pattern = r"""            # Convert test name to test_type expected by the function
            test_name_mapping = \{
                "chi-square": "normal approximation",
                "fisher's exact": "fishers exact",
                "z-test": "normal approximation"
            \}
            mapped_test_type = test_name_mapping\.get\(test\.lower\(\), "normal approximation"\)"""

mde_replacement = """            # Convert test name to test_type expected by the function
            test_name_mapping = {
                "normal approximation": "normal approximation",
                "fisher's exact test": "fishers exact", 
                "likelihood ratio test": "likelihood ratio"
            }
            # Convert the test_type to lowercase for case-insensitive matching
            mapped_test_type = test_name_mapping.get(test_type.lower(), "normal approximation")"""

# Apply the updates
updated_content = re.sub(power_pattern, power_replacement, content)
updated_content = re.sub(mde_pattern, mde_replacement, updated_content)

with open(file_path, 'w') as f:
    f.write(updated_content)

print("Updated test type mappings in parallel_rct.py")
