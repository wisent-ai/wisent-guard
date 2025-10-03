from __future__ import annotations

import io
import json
import os
import tarfile
import shlex
import time
import uuid
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from collections.abc import Iterable

import docker 
from docker.errors import APIError, DockerException, ImageNotFound  
from docker.types import Ulimit  

from wisent_guard.benchmarks.coding.safe_docker.config import SandboxConfig
from wisent_guard.benchmarks.coding.safe_docker.languages import LANGUAGES, LanguageSpec

if TYPE_CHECKING: 
    from docker.models.containers import Container
    from wisent_guard.benchmarks.coding.safe_docker.verifier import Comparator

__all__ = ["DockerSandbox", "TestCase", "TestResult", "RunReport"]


@dataclass(frozen=True, slots=True)
class TestCase:
    """
    A single test case for a coding problem.
    
    arguments:
        name: 
            Name of the test case.
        input_data:
            Input data as a string (e.g., "[1, 2, 3]").
        expected_output:
            Expected output as a string.
        comparator: 
            Comparator to use for output comparison (default: "strip").
        timeout_s: 
            Optional timeout in seconds for this test case (overrides default).

    example:
        TestCase(
            name="simple case",
            input_data="[3, 1, 2]",
            expected_output="[1, 2, 3]",
            comparator="strip",
            timeout_s=2,
        )
    """
    name: str
    input_data: str
    expected_output: str
    comparator: Comparator | str = "strip"
    timeout_s: int | None = None


@dataclass(frozen=True, slots=True)
class TestResult:
    """
    Result of a single test case execution.

    arguments:
        name: 
            Name of the test case.
        passed: 
            Whether the test passed (output matched and exit code 0).
        actual_output: 
            Actual output from the program.
        stderr: 
            Captured stderr output.
        exit_code: 
            Exit code of the program (None if not available).
        duration_s: 
            Execution time in seconds.
        reason: 
            Optional reason for failure if not passed.

    example:
        TestResult(
            name="simple case",
            passed=True,
            actual_output="[1, 2, 3]",
            stderr="",
            exit_code=0,
            duration_s=0.123,
            reason=None,
        )
    """

    name: str
    passed: bool
    actual_output: str
    stderr: str
    exit_code: int | None
    duration_s: float
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RunReport:
    """
    Report of a single run of a submission.

    arguments:
        language:
            The programming language of the submission, e.g., "python".
        compiled:
            Whether the compilation step succeeded (True if no compile step).
        compile_stderr:
            Stderr output from the compilation step (empty if no compile step).
        compile_exit_code:
            Exit code from the compilation step (None if no compile step).
        tests:
            List of TestResult objects for each test case.
    
    example:
        RunReport(
            language="python",
            compiled=True,
            compile_stderr="",
            compile_exit_code=0,
            tests=[TestResult(...), ...],
        )
    """
    language: str
    compiled: bool
    compile_stderr: str
    compile_exit_code: int | None
    tests: list[TestResult]


class DockerSandbox:
    """
    Sandbox runner using Docker containers for isolation.

    arguments:
        config: Optional SandboxConfig to customize resource limits and behavior.
        docker_host: Optional Docker host URL (e.g., "tcp://127.0.0.1:2375").
    """
    def __init__(self, config: SandboxConfig | None = None, docker_host: str | None = None) -> None:
        """
        Initialize the DockerSandbox with optional configuration and Docker host.

        arguments:
            config: 
                Optional SandboxConfig to customize resource limits and behavior.
            docker_host: 
                Optional Docker host URL (e.g., "tcp://127.0.0.1:2375").
        """

        self.cfg = config or SandboxConfig()
        try:
            self.client = docker.DockerClient(base_url=docker_host) if docker_host else docker.from_env()
            self.client.ping()  
        except DockerException as e:
            raise RuntimeError(
                "Could not connect to a Docker daemon. Ensure Docker Engine is installed and running, "
                "or set DOCKER_HOST to a reachable daemon."
            ) from e

    def run_submission(
        self,
        language: str,
        code: str,
        tests: Iterable[TestCase],
        image_override: str | None = None,
        entrypoint_override: str | None = None,
    ) -> RunReport:
        """
        Run a code submission against provided test cases in an isolated Docker container.

        arguments:
            language: The programming language of the submission (e.g., "python").
            code: The source code to run.
            tests: A list of test cases to execute.
            image_override: Optional Docker image to use (defaults to language-specific image).
            entrypoint_override: Optional name of the entrypoint function to call from the wrapper
                (overrides the language spec's wrapper_entrypoint if provided).

        returns:
            RunReport

        raises:
            ValueError: If the specified language is not supported.
            RuntimeError: If there are issues starting the container or running commands.
        
        example:
            >>> runner = DockerSandbox()
            >>> report = runner.run_submission(
            ...     language="python",
            ...     code="def sort_numbers(nums): return sorted(nums)",
            ...     tests=[
            ...         TestCase(
            ...             name="simple case",
            ...             input_data="[3, 1, 2]",
            ...             expected_output="[1, 2, 3]",
            ...             comparator="strip",
            ...             timeout_s=2,
            ...         )
            ...     ],
            ...     entrypoint_override="sort_numbers",
            ... )
            >>> print(report)
            RunReport(
                language="python",
                compiled=True,
                compile_stderr="",
                compile_exit_code=0,
                tests=[
                    TestResult(
                        name="simple case",
                        passed=True,
                        duration=0.1,
                        stdout="[1, 2, 3]",
                        stderr="",
                        exit_code=0,
                    )
                ]
            )
        """
        spec = self._language_spec(language, image_override)
        container: Container | None = None
        compile_err = ""
        compile_code: int | None = None
        compiled_ok = False
        results: list[TestResult] = []
        workspace = "/tmp/sandbox"  

        try:
            container = self._start_container(spec)
            code_to_write = code
            ep_name = entrypoint_override or getattr(spec, "wrapper_entrypoint", None)

            if getattr(spec, "wrapper_template", None) and ep_name:
                try:
                    formatted_wrapper = spec.wrapper_template.replace("{entrypoint}", str(ep_name))
                    wrapper_filename = getattr(spec, "wrapper_filename", None)
                    if isinstance(wrapper_filename, str) and wrapper_filename:
                        self._copy_text(container, workspace, wrapper_filename, formatted_wrapper)
                    else:
                        code_to_write = code_to_write.rstrip() + "\n\n" + formatted_wrapper.lstrip()
                except Exception:
                    pass

            self._copy_text(container, workspace, spec.source_filename, code_to_write)

            if spec.compile_cmd:
                compile_code, compile_out, compile_err, _ = self._exec_with_timeout(
                    container,
                    self._with_timeout(spec.compile_cmd, self.cfg.compile_timeout_s),
                    workdir=workspace,
                )
                compiled_ok = compile_code == 0
                if not compiled_ok:
                    return RunReport(
                        language=language,
                        compiled=False,
                        compile_stderr=compile_out + compile_err,
                        compile_exit_code=compile_code,
                        tests=[],
                    )
            else:
                compiled_ok = True

            for tc in tests:
                input_path = f"{workspace}/input.txt"
                input_text = tc.input_data if isinstance(tc.input_data, str) else json.dumps(tc.input_data)
                self._copy_text(container, workspace, "input.txt", input_text)
                run_cmd = self._with_timeout(
                    spec.run_cmd + ["<", input_path],
                    tc.timeout_s or self.cfg.run_timeout_s,
                )
                code_rc, out, err, dur = self._exec_with_timeout(container, run_cmd, workdir=workspace)

                out_clean = out.rstrip("\r\n")

                passed, reason = self._compare(out_clean, tc.expected_output, tc.comparator)
                results.append(
                    TestResult(
                        name=tc.name,
                        passed=passed and code_rc == 0,
                        actual_output=out_clean,
                        stderr=err,
                        exit_code=code_rc,
                        duration_s=dur,
                        reason=reason if not passed or code_rc != 0 else None,
                    )
                )

            return RunReport(
                language=language,
                compiled=compiled_ok,
                compile_stderr=compile_err,
                compile_exit_code=compile_code,
                tests=results,
            )
        finally:
            if container:
                try:
                    container.remove(force=True)
                except DockerException:
                    pass


    def _language_spec(self, language: str, image_override: str | None) -> LanguageSpec:
        """
        Get the LanguageSpec for the given language, applying any image overrides.
        
        arguments:
            language: 
                The programming language of the code submission.
            image_override:
                An optional Docker image to use instead of the default for the language.

        returns:
            LanguageSpec

        raises:
            ValueError: If the specified language is not supported.
            DockerException: If there is an error communicating with the Docker daemon.

        example:
            >>> runner = DockerSandbox()
            >>> spec = runner._language_spec("python", None)
            >>> print(spec)
            LanguageSpec(
                name="python",
                image="python:3.12-slim",
                source_filename="main.py",
                compile_cmd=None,
                run_cmd=["python", "-u", "main.py"],
                wrapper_template="if __name__ == \"__main__\": ...",
                wrapper_entrypoint="solve",
                wrapper_filename=None,
            )
        """
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
        spec = LANGUAGES[language]
        if self.cfg.image_overrides and language in self.cfg.image_overrides:
            spec = LanguageSpec(**{**spec.__dict__, "image": self.cfg.image_overrides[language]})
        if image_override:
            spec = LanguageSpec(**{**spec.__dict__, "image": image_override})
        return spec

    def _ensure_image(self, image: str) -> None:
        """
        Ensure the specified Docker image is available locally, pulling it if necessary.

        arguments:
            image: The Docker image name (e.g., "python:3.12-slim")
        """
        try:
            self.client.images.get(image)
            return
        except ImageNotFound:
            pass
        repo, tag = image.split(":", 1) if ":" in image else (image, "latest")
        try:
            self.client.images.pull(repository=repo, tag=tag)  # programmatic `docker pull`
        except APIError as e:
            raise RuntimeError(f"Could not pull Docker image '{image}'. Try `docker pull {image}`.") from e

    def _start_container(self, spec: LanguageSpec) -> Container:
        """
        Start a long-lived, hardened container that's guaranteed to be running.

        arguments:
            spec: The LanguageSpec defining the environment to use.
        
        returns:
            A running Docker Container instance.

        raises:
            RuntimeError: If unable to start a running container with the specified image.
            DockerException: If there is an error communicating with the Docker daemon.
        
        example:
            >>> runner = DockerSandbox()
            >>> spec = runner._language_spec("python", None)
            >>> container = runner._start_container(spec)
            >>> print(container.id)
            "e5f6d7c8b9a0..."
        """
        cfg = self.cfg
        self._ensure_image(spec.image)

        ulimits = [
            Ulimit(name="nofile", soft=cfg.nofile_soft, hard=cfg.nofile_hard),
            Ulimit(name="fsize", soft=cfg.fsize_soft_bytes, hard=cfg.fsize_hard_bytes),
        ]

        keeper_cmds: list[list[str]] = [
            ["tail", "-f", "/dev/null"],
            ["sh", "-lc", "sleep infinity"],
            ["sh", "-lc", "while :; do sleep 3600; done"],
        ]

        last_err: Exception | None = None
        for cmd in keeper_cmds:
            try:
                c = self.client.containers.run(
                    image=spec.image,
                    command=cmd,
                    detach=True,                 
                    user=cfg.user,                
                    working_dir="/tmp/sandbox",           
                    cap_drop=["ALL"],
                    security_opt=["no-new-privileges"],
                    read_only=cfg.read_only_root,
                    network_mode=cfg.network_mode, 
                    pids_limit=cfg.pids_limit,
                    nano_cpus=int(cfg.cpus * 1_000_000_000),
                    mem_limit=f"{cfg.memory_mb}m",
                    ulimits=ulimits,
                    # Provide exec-capable tmpfs at the workspace for compiled artifacts
                    tmpfs={"/tmp/sandbox": cfg.tmpfs_opts},  
                    init=True,                      
                )
                c.reload()
                if c.status == "running":
                    return c
                logs = (c.logs(tail=50) or b"").decode("utf-8", "ignore")
                c.remove(force=True)
                last_err = RuntimeError(f"Container exited immediately. Logs:\n{logs}")
            except APIError as e:
                last_err = e
                continue

        raise RuntimeError("Failed to start a running container with any keeper command") from last_err

    def _ensure_dir(self, container: Container, path: str, mode: int = 0o755) -> None:
        """
        Ensure 'path' exists inside the container.
        Creates it by uploading a tar directory entry to its parent (which must exist).

        arguments:
            container:
                The Docker container instance.
            path:
                The path to the directory to ensure.
            mode:
                The permissions mode to set for the directory.

        raises:
            RuntimeError: If unable to create the directory inside the container.            
        """
        parent, leaf = os.path.split(path.rstrip("/"))
        parent = parent or "/"
        if not leaf:
            return  

        dir_tar = io.BytesIO()
        with tarfile.open(fileobj=dir_tar, mode="w") as tar:
            info = tarfile.TarInfo(name=f"{leaf}/")
            info.type = tarfile.DIRTYPE
            info.mode = mode
            tar.addfile(info)
        dir_tar.seek(0)
        try:
            container.put_archive(parent, dir_tar.getvalue())  # parent exists (/tmp)
        except APIError as e:
            raise RuntimeError(f"Failed to create directory '{path}' inside container") from e

        res = container.exec_run(["sh", "-lc", f"test -d {path}"])
        if res.exit_code != 0:
            out = res.output if isinstance(res.output, (bytes, bytearray)) else b""
            msg = out.decode("utf-8", "ignore")
            raise RuntimeError(f"Workspace path '{path}' not present after creation attempt. Details: {msg}")

    @staticmethod
    def _copy_text(container: Container, folder: str, filename: str, text: str) -> None:
        """
        Write a text file inside the container without using put_archive (which
        can be blocked when rootfs is read-only). We rely on a writable tmpfs
        mounted at 'folder' and create the file via a shell here-doc.

        arguments:
            container:
                The Docker container instance.
            folder:
                The folder inside the container where the file will be created.
            filename:
                The name of the file to create inside `folder`.
            text:
                The text content to write to the file.
        
        raises:
            RuntimeError: If unable to write the file inside the container.

        example:
            >>> runner = DockerSandbox()
            >>> container = runner._start_container(runner._language_spec("python", None))
            >>> runner._copy_text(container, "/tmp/sandbox", "main.py", "print('Hello, World!')")
            >>> res = container.exec_run(["cat", "/tmp/sandbox/main.py"])
            >>> print(res.output.decode("utf-8"))
            print('Hello, World!')
        """
        try:
            container.exec_run(["sh", "-lc", f"mkdir -p {shlex.quote(folder)}"], workdir="/")
        except Exception:
            pass

        target = os.path.join(folder, filename)
        delimiter = f"EOF__WG__{uuid.uuid4().hex}"
        heredoc_cmd = (
            f"cat > {shlex.quote(target)} << '{delimiter}'\n" + text + f"\n{delimiter}\n"
        )
        res = container.exec_run(["sh", "-lc", heredoc_cmd], workdir=folder)
        if res.exit_code != 0:
            out = res.output if isinstance(res.output, (bytes, bytearray)) else b""
            msg = out.decode("utf-8", "ignore")
            raise RuntimeError(f"Failed to write file '{target}' in container. Details: {msg}")

    @staticmethod
    def _shell_join_with_redirection(argv: list[str]) -> str:
        """Safely join argv into a shell command string, allowing redirection tokens.

        We quote all arguments except redirection operators themselves so that
        input/output redirection works as intended without exposing shell injection.
        Supported operators: <, >, >>, 2>, 2>>.

        arguments:
            argv: 
                List of command arguments, including any redirection tokens.

        returns:
            A safely quoted shell command string.
        
        example:
            >>> cmd = ["python", "script.py", "<", "input.txt", ">", "output.txt"]
            >>> safe_cmd = DockerSandbox._shell_join_with_redirection(cmd)
            >>> print(safe_cmd)
            python script.py < input.txt > output.txt
        """
        redirs = {"<", ">", ">>", "2>", "2>>"}
        out: list[str] = []
        i = 0
        n = len(argv)
        while i < n:
            tok = argv[i]
            if tok in redirs and i + 1 < n:
                out.append(tok)
                out.append(shlex.quote(argv[i + 1]))
                i += 2
            else:
                out.append(shlex.quote(tok))
                i += 1
        return " ".join(out)

    def _exec_with_timeout(
        self, container: Container, cmd: list[str], workdir: str
    ) -> tuple[int, str, str, float]:
        """
        Run a command and return (exit_code, stdout, stderr, duration_s).
        Uses in-container JSON trailer for timing; falls back to host perf_counter.

        arguments:
            container:
                The Docker container instance.
            cmd:
                The command to execute as a list of arguments.
            workdir:
                The working directory inside the container to run the command in.

        returns:
            A tuple of (exit_code, stdout, stderr, duration_s).

        raises:
            RuntimeError: If there is an error executing the command inside the container.
            DockerException: If there is an error communicating with the Docker daemon.

        example:
            >>> runner = DockerSandbox()
            >>> container = runner._start_container(runner._language_spec("python", None))
            >>> rc, out, err, dur = runner._exec_with_timeout(container, ["python", "-c", "print('Hello')"], "/tmp/sandbox")
            >>> print(rc, out, err, dur)
            0 Hello  0.05
        """
        cmd_str = self._shell_join_with_redirection(cmd)
        wrapped = [
            "sh",
            "-lc",
            (
                "start=$(date +%s%3N); "
                + cmd_str
                + r"; rc=$?; end=$(date +%s%3N); "
                + r'echo "__TIMER__:{\"ms\":$((end-start)),\"rc\":$rc}" 1>&2; '
                + "exit $rc"
            ),
        ]

        host_t0 = time.perf_counter()
        result = container.exec_run(wrapped, workdir=workdir, demux=True)  
        host_duration = time.perf_counter() - host_t0

        stdout_b, stderr_b = result.output if isinstance(result.output, tuple) else (result.output, b"")
        stdout = (stdout_b or b"").decode("utf-8", errors="replace")
        stderr = (stderr_b or b"").decode("utf-8", errors="replace")
        rc = result.exit_code

        duration_s = host_duration
        marker = "__TIMER__:{"
        if marker in stderr:
            prefix, trailer = stderr.split(marker, 1)
            json_part = "{" + trailer.split("}", 1)[0] + "}"
            try:
                meta = json.loads(json_part)
                duration_s = float(meta.get("ms", 0.0)) / 1000.0
                rc = meta.get("rc", rc)
                stderr = prefix
            except Exception:
                pass

        return (rc if rc is not None else -1), stdout, stderr, duration_s

    @staticmethod
    def _with_timeout(cmd: list[str], timeout_s: int) -> list[str]:
        """
        Wrap the command with GNU 'timeout' to enforce a time limit.
        arguments:
            cmd: 
                The command to execute as a list of arguments.
            timeout_s: 
                The timeout in seconds.

        returns:
            A new command list wrapped with 'timeout'.

        example:
            >>> cmd = ["python", "script.py"]
            >>> timed_cmd = DockerSandbox._with_timeout(cmd, 5)
            >>> print(timed_cmd)
            ['timeout', '-s', 'KILL', '5', 'python', 'script.py']
        """
        return ["timeout", "-s", "KILL", str(timeout_s)] + cmd

    @staticmethod
    def _compare(actual: str, expected: Any, comparator: Comparator | str) -> tuple[bool, str | None]:
        """
        Compare actual output to expected using the specified comparator.

        arguments:
            actual: 
                The actual output string from the program.
            expected: 
                The expected output, either as a string or a JSON-serializable object.
            comparator: 
                The comparator to use, either as a Comparator object or a string name.

        returns:
            A tuple of (passed: bool, reason: str | None).

        raises:
            ValueError: If the comparator is not recognized.

        example:
            >>> passed, reason = DockerSandbox._compare(" 42 ", "42", "strip")
            >>> print(passed, reason)
            True None
        """
        actual_norm = (actual or "").replace("\r\n", "\n")
        exp_str = expected if isinstance(expected, str) else json.dumps(expected)

        comp_name = comparator.name if hasattr(comparator, "name") else comparator
        comp_func = getattr(comparator, "func", None)

        ok = False
        if callable(comp_func):
            try:
                ok = bool(comp_func(actual_norm, exp_str))
            except Exception:
                ok = False
        else:
            match comp_name:
                case "exact":
                    ok = actual_norm == exp_str
                case "strip":
                    ok = actual_norm.strip() == exp_str.strip()
                case "regex":
                    import re
                    ok = re.fullmatch(exp_str, actual_norm) is not None
                case _:
                    ok = actual_norm.strip() == exp_str.strip()

        reason = None if ok else f"Comparator '{comp_name}' failed."
        return ok, reason