from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

__all__ = ["SandboxConfig"]

@dataclass(frozen=True)
class SandboxConfig:
    """
    Immutable configuration for sandboxed runs.

    attributes:
        memory_mb: Maximum memory in megabytes.
        cpus: Number of CPU cores to allocate.
        pids_limit: Maximum number of process IDs.
        nproc_soft: Soft limit on number of processes.
        nproc_hard: Hard limit on number of processes.
        nofile_soft: Soft limit on number of open files.
        nofile_hard: Hard limit on number of open files.
        fsize_soft_bytes: Soft limit on file size in bytes.
        fsize_hard_bytes: Hard limit on file size in bytes.
        read_only_root: If True, mounts the root filesystem as read-only.
        network_mode: Network mode for the container (e.g., "none", "bridge").
        user: User and group ID to run the container as (e.g., "1000:1000").
        tmpfs_opts: Mount options for /tmp (e.g., "nosuid,nodev,mode=1777").
        sandbox_tmpfs_opts: Mount options for sandbox directories (e.g., "nosuid,nodev,mode=0755").
        compile_timeout_s: Timeout in seconds for compilation steps.
        run_timeout_s: Timeout in seconds for execution steps.
        image_overrides: Optional mapping to override default Docker images for specific languages.

    note:
        By default we don't allow any network access and run with a non-root user.
    """
    memory_mb: int = 512
    cpus: float = 1.0  
    pids_limit: int = 256
    nproc_soft: int = 128
    nproc_hard: int = 128
    nofile_soft: int = 256
    nofile_hard: int = 256
    fsize_soft_bytes: int = 20 * 1024 * 1024  
    fsize_hard_bytes: int = 20 * 1024 * 1024
    read_only_root: bool = True
    network_mode: str = "none" 
    user: str = "1000:1000"  
    tmpfs_opts: str = "nosuid,nodev,exec,mode=1777"
    sandbox_tmpfs_opts: str = "nosuid,nodev,mode=0755"  
    compile_timeout_s: int = 15
    run_timeout_s: int = 3

    image_overrides: Mapping[str, str] | None = None
