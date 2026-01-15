#!/usr/bin/env python3
"""Debug answer extraction."""

import re

test_string1 = "The total is \\boxed{18} dollars."
test_string2 = r"The total is \boxed{18} dollars."

print(f"String 1: {repr(test_string1)}")
print(f"String 2: {repr(test_string2)}")
print()

# Test basic pattern
pattern = r"\\boxed\{(\-?[0-9\.\,]+)\}"
print(f"Pattern: {repr(pattern)}")

match1 = re.search(pattern, test_string1)
match2 = re.search(pattern, test_string2)

print(f"Match string 1: {match1.group(1) if match1 else 'No match'}")
print(f"Match string 2: {match2.group(1) if match2 else 'No match'}")