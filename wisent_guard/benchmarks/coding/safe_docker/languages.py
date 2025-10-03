from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LanguageSpec", "LANGUAGES"]

@dataclass(frozen=True)
class LanguageSpec:
    """
    Specification for a programming language environment.
    
    attributes:
        name: Name of the programming language.
        image: Docker image to use for this language.
        source_filename: Filename to save the source code as.
        compile_cmd: Command to compile the source code (None if interpreted).
        run_cmd: Command to execute the compiled/interpreted code.
        wrapper_template: Optional code snippet appended to the submission to wrap stdin->entrypoint->stdout.
            Use Python-style format placeholder {entrypoint} to reference the configured entrypoint symbol.
            Example (Python):
                if __name__ == "__main__":
                    import sys, json
                    _in = sys.stdin.read()
                    data = json.loads(_in) if _in.strip() else None
                    res = {entrypoint}(data)
                    sys.stdout.write(json.dumps(res))
        wrapper_entrypoint: Optional single entrypoint symbol to substitute into wrapper_template's {entrypoint}.
        wrapper_filename: Optional filename for the wrapper. If provided, wrapper is written as a separate file
            (useful for languages like Java or C++ where separate files are required). If omitted, wrapper is appended
            to the same source file.
    """
    name: str
    image: str
    source_filename: str
    compile_cmd: list[str] | None
    run_cmd: list[str]
    wrapper_template: str | None = None
    wrapper_entrypoint: str | None = None
    wrapper_filename: str | None = None


LANGUAGES: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        name="python",
        image="python:3.12-slim",                
        source_filename="main.py",
        compile_cmd=None,
        run_cmd=["python", "-u", "main.py"],
        wrapper_template=(
            "if __name__ == \"__main__\":\n"
            "    import sys, json\n"
            "    _in = sys.stdin.read()\n"
            "    fn = globals().get(\"{entrypoint}\")\n"
            "    if callable(fn):\n"
            "        data = json.loads(_in) if _in.strip() else None\n"
            "        res = fn(data)\n"
            "        out = json.dumps(res)\n"
            "    else:\n"
            "        out = _in\n"
            "    sys.stdout.write(out)\n"
        ),
        wrapper_entrypoint="solve",
    ),
    "cpp": LanguageSpec(
        name="cpp",
        image="gcc:14-bookworm",                
        source_filename="main.cpp",
        compile_cmd=["bash", "-lc", "g++ -std=c++20 -O2 -pipe -o main main.cpp wrapper.cpp"],
        run_cmd=["bash", "-lc", "./main"],
        wrapper_template=(
            "#include <bits/stdc++.h>\n"
            "using namespace std;\n"
            "extern std::string {entrypoint}(const std::string&);\n"
            "int main(){\n"
            "    std::ios::sync_with_stdio(false); cin.tie(nullptr);\n"
            "    std::string input((std::istreambuf_iterator<char>(cin)), std::istreambuf_iterator<char>());\n"
            "    auto out = {entrypoint}(input);\n"
            "    cout<<out;\n"
            "    return 0;\n"
            "}\n"
        ),
        wrapper_entrypoint="solve",
        wrapper_filename="wrapper.cpp",
    ),
    "java": LanguageSpec(
        name="java",
        image="eclipse-temurin:21-jdk",          
        source_filename="Main.java",
        compile_cmd=["bash", "-lc", "javac Main.java Wrapper.java"],
        run_cmd=["bash", "-lc", "java -Xss64m -Xmx256m Wrapper"],
        wrapper_template=(
            "import java.io.*;\n"
            "import java.nio.charset.StandardCharsets;\n"
            "class Wrapper {\n"
            "  public static void main(String[] args) throws Exception {\n"
            "    var in = new String(System.in.readAllBytes(), StandardCharsets.UTF_8);\n"
            "    var out = Main.{entrypoint}(in);\n"
            "    System.out.print(out);\n"
            "  }\n"
            "}\n"
        ),
        wrapper_entrypoint="solve",
        wrapper_filename="Wrapper.java",
    ),
}
