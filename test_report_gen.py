#!/usr/bin/env python3
from core.utils.report_generator import generate_sample_size_report

# Test parameters
results = {
    'n1': 50,
    'n2': 50,
    'effect_size': 0.5
}

params = {
    'mean1': 0,
    'mean2': 0.5,
    'std_dev': 1.0,
    'power': 0.8,
    'alpha': 0.05,
    'hypothesis_type': 'Superiority',
    'method': 'analytical'
}

# Generate report
report = generate_sample_size_report(results, params, 'Parallel RCT', 'Continuous Outcome')

# Save to file for inspection
with open('test_report_output.html', 'w') as f:
    f.write(report)

print("Report generated and saved to test_report_output.html")
print("\nFirst 500 characters:")
print(report[:500])
print("\nSearching for the tip section:")
tip_index = report.find("Tip:")
if tip_index > 0:
    print(report[tip_index-100:tip_index+200])