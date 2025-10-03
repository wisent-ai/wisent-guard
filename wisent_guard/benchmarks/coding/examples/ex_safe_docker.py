# Example of safe Docker usage

from wisent_guard.benchmarks.coding.safe_docker.runner import DockerSandbox, TestCase
from wisent_guard.benchmarks.coding.safe_docker.verifier import EXACT

# Create a sandbox with default configuration (see config.py for options)
sandbox = DockerSandbox()

# Python sorting example

language = "python" 

code_bad = """
def solve(nums):
    # A bad sorting algorithm (does not sort correctly)
    return nums
"""

code_suboptimal = """
def solve(nums):
    # A suboptimal sorting algorithm (Bubble Sort)
    n = len(nums)
    for i in range(n):
        for j in range(0, n-i-1):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
    return nums
"""

code_good = """
def solve(nums):
    # A good sorting algorithm (Timsort, used by Python's built-in sort)
    return sorted(nums)
"""

# Define test cases
test_cases = [
    TestCase(
        name="Basic test",
        input_data='[5, 3, 8, 1]',
        expected_output='[1, 3, 5, 8]',
        comparator=EXACT,
    ),
    TestCase(
        name="Another basic test",
        input_data='[3, 1, 2]',
        expected_output='[1, 2, 3]',
        comparator=EXACT,
    ),
    TestCase(
        name="Already sorted test",
        input_data='[1, 2, 3]',
        expected_output='[1, 2, 3]',
        comparator=EXACT,
    ),
    TestCase(
        name="Empty list test",
        input_data='[]',
        expected_output='[]',
        comparator=EXACT,
    ),
]

# Run the bad code
print("Testing bad code:")
report_bad = sandbox.run_submission(language, code_bad, test_cases)
print(report_bad)

# Run the suboptimal code
print("\nTesting suboptimal code:")
report_suboptimal = sandbox.run_submission(language, code_suboptimal, test_cases)
print(report_suboptimal)

# Run the good code
print("\nTesting good code:")
report_good = sandbox.run_submission(language, code_good, test_cases)
print(report_good)

# We can also test different coding styles by adjusting the wrapper or entrypoint

# Custom entrypoint name using entrypoint_override
code_custom_entrypoint = """
def sort_numbers(nums):
    return sorted(nums)
"""

print("\nTesting custom entrypoint (sort_numbers) via override:")
report_custom = sandbox.run_submission(
    language,
    code_custom_entrypoint,
    test_cases,
    entrypoint_override="sort_numbers",
)
print(report_custom)

# Class-based implementation with a thin solve adapter
code_class_based = """
class Sorter:
    def sort(self, nums):
        return sorted(nums)

def solve(nums):
    return Sorter().sort(nums)
"""

print("\nTesting class-based solution (solve calls Sorter):")
report_class = sandbox.run_submission(language, code_class_based, test_cases)
print(report_class)

# Direct I/O script (reads stdin, writes stdout) — disable wrapper
code_direct_io = """
import sys, json
nums = json.loads(sys.stdin.read())
sys.stdout.write(json.dumps(sorted(nums)))
"""

print("\nTesting direct I/O script (no function; wrapper disabled):")
report_direct = sandbox.run_submission(
    language,
    code_direct_io,
    test_cases,
    entrypoint_override="", 
)
print(report_direct)

print("End of Python examples.")

# C++ example (sorting)

code_cpp = r"""
#include <bits/stdc++.h>
using namespace std;

// Expected by the wrapper: std::string solve(const std::string&)
std::string solve(const std::string& input) {
    vector<long long> nums;
    long long num = 0;
    int sign = 1;
    bool in_num = false;
    for (char c : input) {
        if (c == '-') { sign = -1; }
        else if (c >= '0' && c <= '9') {
            in_num = true;
            num = num * 10 + (c - '0');
        } else {
            if (in_num) { nums.push_back(sign * num); num = 0; sign = 1; in_num = false; }
        }
    }
    if (in_num) nums.push_back(sign * num);

    sort(nums.begin(), nums.end());

    // Build output like: [1, 2, 3]
    ostringstream out;
    out << '[';
    for (size_t i = 0; i < nums.size(); ++i) {
        if (i) out << ", ";
        out << nums[i];
    }
    out << ']';
    return out.str();
}
"""

print("\nTesting C++ code:")
report_cpp = sandbox.run_submission("cpp", code_cpp, test_cases)
print(report_cpp)

# Java example (sorting)

code_java = r"""
import java.util.*;

public class Main {
    // Expected by the wrapper: static String solve(String input)
    public static String solve(String input) {
        List<Long> nums = new ArrayList<>();
        long num = 0; int sign = 1; boolean inNum = false;
        for (int i = 0; i < input.length(); i++) {
            char c = input.charAt(i);
            if (c == '-') { sign = -1; }
            else if (c >= '0' && c <= '9') { inNum = true; num = num * 10 + (c - '0'); }
            else {
                if (inNum) { nums.add(sign * num); num = 0; sign = 1; inNum = false; }
            }
        }
        if (inNum) nums.add(sign * num);

        Collections.sort(nums);
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for (int i = 0; i < nums.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(nums.get(i));
        }
        sb.append(']');
        return sb.toString();
    }
}
"""

print("\nTesting Java code:")
report_java = sandbox.run_submission("java", code_java, test_cases)
print(report_java)