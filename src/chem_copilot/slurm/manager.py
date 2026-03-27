"""Slurm job manager for submitting and monitoring calculations.

Workflow (matches lab cluster setup):
1. Create job directory on worker's /scratch via SSH
2. SCP input files to worker
3. Build control.in (header + species defaults) on the worker
4. Submit sbatch with --chdir pointing to worker scratch dir
5. Monitor with squeue
6. Retrieve output via SSH/SCP

Configuration via environment variables (never hardcoded):
    CHEM_COPILOT_JOBS_DIR       — local metadata directory (default: ~/chem-copilot-jobs)
    CHEM_COPILOT_SCRATCH_DIR    — scratch dir on workers (default: /scratch/chem-copilot)
    CHEM_COPILOT_WORKER         — default worker SSH target (default from SLURM)
    FHIAIMS_BINARY              — path to FHI-aims binary (on workers)
    FHIAIMS_SPECIES_DIR         — path to species_defaults/defaults_2020/ (on workers)
    PSI4_ACTIVATE_SCRIPT        — psi4 activation script path (on workers)
    SLURM_PARTITION             — Slurm partition (default: cpu)
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class JobInfo:
    """Status information for a Slurm job."""
    job_id: str
    name: str = ""
    state: str = "UNKNOWN"
    node: str = ""
    elapsed: str = ""
    exit_code: str = ""
    job_dir: str = ""
    output_file: str = ""


@dataclass
class SlurmConfig:
    """Slurm submission configuration."""
    partition: str = "cpu"
    cpus_per_task: int = 16
    mem: str = "32G"
    time: str = "4:00:00"
    gres: str = ""

    @classmethod
    def from_env(cls) -> SlurmConfig:
        return cls(partition=os.environ.get("SLURM_PARTITION", "cpu"))


class SlurmManager:
    """Manage Slurm job submission and monitoring for the lab cluster."""

    def __init__(
        self,
        jobs_dir: str | Path | None = None,
        scratch_dir: str | None = None,
        fhiaims_binary: str | None = None,
        fhiaims_species_dir: str | None = None,
        psi4_activate_script: str | None = None,
    ):
        # Local metadata directory
        self.jobs_dir = Path(
            jobs_dir
            or os.environ.get("CHEM_COPILOT_JOBS_DIR")
            or Path.home() / "chem-copilot-jobs"
        )
        # Scratch directory on workers
        self.scratch_dir = (
            scratch_dir
            or os.environ.get("CHEM_COPILOT_SCRATCH_DIR", "/scratch/chem-copilot")
        )
        # Paths that exist on the WORKER nodes
        self.fhiaims_binary = (
            fhiaims_binary
            or os.environ.get("FHIAIMS_BINARY")
            or "/home/albd/projects/fhiaims/build/aims.x"
        )
        self.fhiaims_species_dir = (
            fhiaims_species_dir
            or os.environ.get("FHIAIMS_SPECIES_DIR")
            or "/home/albd/projects/fhiaims/species_defaults/defaults_2020"
        )
        self.psi4_activate_script = (
            psi4_activate_script
            or os.environ.get("PSI4_ACTIVATE_SCRIPT")
            or "/home/albd/projects/activate_xdm.sh"
        )
        self.slurm_config = SlurmConfig.from_env()

    # ------------------------------------------------------------------
    # FHI-aims submission
    # ------------------------------------------------------------------

    def submit_fhiaims(
        self,
        control_in: str,
        geometry_in: str,
        job_name: str = "aims",
        basis_tier: str = "light",
        slurm_overrides: dict[str, str] | None = None,
    ) -> str:
        """Submit an FHI-aims calculation via Slurm.

        Creates files on the worker's /scratch, builds control.in with
        species defaults on the worker, and submits with sbatch --chdir.

        Returns the Slurm job ID.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_label = f"{job_name}_{timestamp}"

        # Remote job directory on the worker's scratch
        remote_dir = f"{self.scratch_dir}/{job_label}"

        # Species files to append to control.in
        elements = self._extract_elements(geometry_in)
        species_files = self._species_file_paths(elements, basis_tier)

        # Local metadata directory
        local_dir = self.jobs_dir / job_label
        local_dir.mkdir(parents=True, exist_ok=True)

        # Write files locally
        (local_dir / "control.in.header").write_text(control_in)
        (local_dir / "geometry.in").write_text(geometry_in)
        (local_dir / ".remote_dir").write_text(remote_dir)
        (local_dir / ".program").write_text("fhi-aims")

        # Generate full setup+run script
        cfg = self._merge_config(slurm_overrides)
        full_script = self._fhiaims_full_script(
            remote_dir, job_name, cfg, control_in, geometry_in, species_files,
        )
        (local_dir / "run.sh").write_text(full_script)

        # Submit
        job_id = self._sbatch_file(local_dir / "run.sh")
        (local_dir / ".job_id").write_text(job_id)

        logger.info("Submitted FHI-aims job %s → %s", job_id, remote_dir)
        return job_id

    def submit_psi4(
        self,
        input_script: str,
        job_name: str = "psi4",
        psi4_env: str = "xdm",
        slurm_overrides: dict[str, str] | None = None,
    ) -> str:
        """Submit a Psi4 calculation via Slurm."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_label = f"{job_name}_{timestamp}"
        remote_dir = f"{self.scratch_dir}/{job_label}"

        local_dir = self.jobs_dir / job_label
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "input.py").write_text(input_script)
        (local_dir / ".remote_dir").write_text(remote_dir)
        (local_dir / ".program").write_text("psi4")

        activate = self.psi4_activate_script
        if psi4_env == "apbe0":
            activate = activate.replace("activate_xdm", "activate_apbe0")

        cfg = self._merge_config(slurm_overrides)
        if not slurm_overrides or "partition" not in slurm_overrides:
            cfg.partition = "cpu"

        script = self._psi4_full_script(
            remote_dir, job_name, cfg, input_script, activate,
        )
        script_path = local_dir / "run.sh"
        script_path.write_text(script)

        job_id = self._sbatch_file(script_path)
        (local_dir / ".job_id").write_text(job_id)

        logger.info("Submitted Psi4 job %s → %s", job_id, remote_dir)
        return job_id

    # ------------------------------------------------------------------
    # Job monitoring
    # ------------------------------------------------------------------

    def get_job_status(self, job_id: str) -> JobInfo:
        """Get the status of a Slurm job."""
        info = JobInfo(job_id=job_id)

        # Try squeue (running/pending)
        try:
            out = self._run_cmd(
                ["squeue", "-j", job_id, "--noheader", "--format=%i|%j|%T|%N|%M"],
            )
            if out.strip():
                parts = out.strip().split("|")
                if len(parts) >= 5:
                    info.name = parts[1].strip()
                    info.state = parts[2].strip()
                    info.node = parts[3].strip()
                    info.elapsed = parts[4].strip()
                    info.job_dir = self._find_local_dir(job_id)
                    return info
        except subprocess.CalledProcessError:
            pass

        # Try sacct (completed/failed)
        try:
            out = self._run_cmd(
                ["sacct", "-j", job_id, "--noheader", "--parsable2",
                 "--format=JobID,JobName,State,NodeList,Elapsed,ExitCode"],
            )
            for line in out.strip().splitlines():
                parts = line.split("|")
                if len(parts) >= 6 and parts[0].strip() == job_id:
                    info.name = parts[1].strip()
                    info.state = parts[2].strip()
                    info.node = parts[3].strip()
                    info.elapsed = parts[4].strip()
                    info.exit_code = parts[5].strip()
                    break
        except subprocess.CalledProcessError:
            # sacct may not be available (accounting disabled)
            # Check if output file exists as a proxy for completion
            pass

        info.job_dir = self._find_local_dir(job_id)
        return info

    def list_recent_jobs(self, n: int = 10) -> list[JobInfo]:
        """List recent jobs from the local metadata directory."""
        if not self.jobs_dir.exists():
            return []
        dirs = sorted(
            (d for d in self.jobs_dir.iterdir() if d.is_dir() and (d / ".job_id").exists()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        jobs: list[JobInfo] = []
        for d in dirs[:n]:
            job_id = (d / ".job_id").read_text().strip()
            info = self.get_job_status(job_id)
            info.job_dir = str(d)
            program = (d / ".program").read_text().strip() if (d / ".program").exists() else ""
            if program:
                info.name = f"{program}:{info.name}" if info.name else program
            jobs.append(info)
        return jobs

    def cancel_job(self, job_id: str) -> str:
        """Cancel a Slurm job."""
        try:
            self._run_cmd(["scancel", job_id])
            return f"Job {job_id} cancelled."
        except subprocess.CalledProcessError as e:
            return f"Failed to cancel job {job_id}: {e}"

    # ------------------------------------------------------------------
    # Result retrieval
    # ------------------------------------------------------------------

    def get_output_path(self, job_id: str) -> Path | None:
        """Find the output file for a completed job.

        Checks the local metadata dir for Slurm output, then tries
        the remote scratch dir via SSH if needed.
        """
        local_dir = self._find_local_dir(job_id)
        if not local_dir:
            return None

        d = Path(local_dir)

        # Check local candidates
        for name in [f"aims-{job_id}.out", f"slurm-{job_id}.out", "aims.out", "psi4.out"]:
            p = d / name
            if p.exists() and p.stat().st_size > 0:
                return p

        # Try to fetch from remote
        remote_dir_file = d / ".remote_dir"
        if remote_dir_file.exists():
            remote_dir = remote_dir_file.read_text().strip()
            self._fetch_output(job_id, remote_dir, d)
            # Check again after fetch
            for name in [f"aims-{job_id}.out", "aims.out", "psi4.out", f"slurm-{job_id}.out"]:
                p = d / name
                if p.exists() and p.stat().st_size > 0:
                    return p

        return None

    def get_program(self, job_id: str) -> str:
        local_dir = self._find_local_dir(job_id)
        if not local_dir:
            return "unknown"
        pf = Path(local_dir) / ".program"
        return pf.read_text().strip() if pf.exists() else "unknown"

    # ------------------------------------------------------------------
    # Internal: sbatch scripts
    # ------------------------------------------------------------------

    def _fhiaims_full_script(
        self,
        remote_dir: str,
        job_name: str,
        cfg: SlurmConfig,
        control_in: str,
        geometry_in: str,
        species_files: list[str],
    ) -> str:
        """Full sbatch script that sets up files + runs FHI-aims on the worker."""
        # FHI-aims uses MPI: --ntasks = MPI ranks, --cpus-per-task=1 (pure MPI)
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={cfg.partition}",
            f"#SBATCH --nodes=1",
            f"#SBATCH --ntasks={cfg.cpus_per_task}",
            f"#SBATCH --cpus-per-task=1",
            f"#SBATCH --mem={cfg.mem}",
            f"#SBATCH --time={cfg.time}",
            f"#SBATCH --output={remote_dir}/aims-%j.out",
            f"#SBATCH --error={remote_dir}/aims-%j.out",
        ]
        if cfg.gres:
            lines.append(f"#SBATCH --gres={cfg.gres}")

        species_cat = " ".join(species_files)

        lines.extend([
            "",
            "# FHI-aims requires unlimited stack size",
            "ulimit -s unlimited",
            "",
            f'echo "Running FHI-aims on $(hostname) with $SLURM_NTASKS MPI ranks"',
            "",
            f"# Create working directory",
            f"mkdir -p {remote_dir}",
            f"cd {remote_dir}",
            "",
            "# Write input files",
            f"cat > control.in.header << 'CTRL_EOF'",
            control_in.rstrip(),
            "CTRL_EOF",
            "",
            f"cat > geometry.in << 'GEO_EOF'",
            geometry_in.rstrip(),
            "GEO_EOF",
            "",
            "# Build control.in with species defaults",
            f"cat control.in.header {species_cat} > control.in",
            "",
            "# Verify input files",
            'ls control.in geometry.in || { echo "Missing input files!"; exit 1; }',
            "",
            "# Run FHI-aims",
            "export OMP_NUM_THREADS=1",
            f"mpirun -np $SLURM_NTASKS {self.fhiaims_binary}",
        ])
        return "\n".join(lines) + "\n"

    def _psi4_full_script(
        self,
        remote_dir: str,
        job_name: str,
        cfg: SlurmConfig,
        input_script: str,
        activate_script: str,
    ) -> str:
        """Full sbatch script that sets up files + runs Psi4 on the worker."""
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={cfg.partition}",
            f"#SBATCH --nodes=1",
            f"#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={cfg.cpus_per_task}",
            f"#SBATCH --mem={cfg.mem}",
            f"#SBATCH --time={cfg.time}",
            f"#SBATCH --output={remote_dir}/slurm-%j.out",
            f"#SBATCH --error={remote_dir}/slurm-%j.out",
            "",
            f"mkdir -p {remote_dir}",
            f"cd {remote_dir}",
            "",
            f"cat > input.py << 'PSI4_EOF'",
            input_script.rstrip(),
            "PSI4_EOF",
            "",
            f"source {activate_script}",
            f"export OMP_NUM_THREADS={cfg.cpus_per_task}",
            f"export PSI_SCRATCH={remote_dir}",
            "",
            "python input.py > psi4.out 2>&1",
        ]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Internal: species defaults
    # ------------------------------------------------------------------

    def _species_file_paths(self, elements: list[str], tier: str) -> list[str]:
        """Return full paths to species default files (on the worker)."""
        from chem_copilot.generators.fhi_aims import _ATOMIC_NUMBERS
        paths = []
        for elem in elements:
            z = _ATOMIC_NUMBERS.get(elem, 0)
            paths.append(f"{self.fhiaims_species_dir}/{tier}/{z:02d}_{elem}_default")
        return paths

    @staticmethod
    def _extract_elements(geometry_in: str) -> list[str]:
        """Extract unique element symbols from a geometry.in file."""
        seen: set[str] = set()
        result: list[str] = []
        for line in geometry_in.splitlines():
            line = line.strip()
            if line.startswith(("atom ", "atom_frac ")):
                parts = line.split()
                if len(parts) >= 5:
                    elem = parts[4]
                    if elem not in seen:
                        seen.add(elem)
                        result.append(elem)
        return result

    # ------------------------------------------------------------------
    # Internal: output fetching
    # ------------------------------------------------------------------

    def _fetch_output(self, job_id: str, remote_dir: str, local_dir: Path) -> None:
        """Try to copy output files from the remote worker to local dir."""
        # Determine which node ran the job
        info = self.get_job_status(job_id)
        node = info.node
        if not node:
            # Try all known workers
            for candidate in ["odin", "loki", "freya"]:
                try:
                    self._run_cmd(
                        ["scp", "-q", f"albd@{candidate}:{remote_dir}/aims-{job_id}.out",
                         str(local_dir / f"aims-{job_id}.out")],
                        timeout=10,
                    )
                    return
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    continue
            return

        # Fetch from the known node
        for pattern in [f"aims-{job_id}.out", "psi4.out"]:
            try:
                self._run_cmd(
                    ["scp", "-q", f"albd@{node}:{remote_dir}/{pattern}",
                     str(local_dir / pattern)],
                    timeout=15,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass

    # ------------------------------------------------------------------
    # Internal: subprocess helpers
    # ------------------------------------------------------------------

    def _find_local_dir(self, job_id: str) -> str:
        """Find the local metadata directory for a given job ID."""
        if not self.jobs_dir.exists():
            return ""
        for d in self.jobs_dir.iterdir():
            if d.is_dir() and (d / ".job_id").exists():
                if (d / ".job_id").read_text().strip() == job_id:
                    return str(d)
        return ""

    def _merge_config(self, overrides: dict[str, str] | None) -> SlurmConfig:
        cfg = SlurmConfig(partition=self.slurm_config.partition)
        if overrides:
            for key, val in overrides.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, type(getattr(cfg, key))(val) if not isinstance(val, str) else val)
                    # Handle int fields
                    if key == "cpus_per_task":
                        cfg.cpus_per_task = int(val)
        return cfg

    @staticmethod
    def _sbatch_file(script_path: Path) -> str:
        """Submit a script via sbatch and return the job ID."""
        out = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True, text=True, check=True,
        )
        match = re.search(r"(\d+)", out.stdout)
        if not match:
            raise RuntimeError(f"Could not parse job ID from sbatch: {out.stdout}")
        return match.group(1)

    @staticmethod
    def _run_cmd(cmd: list[str], timeout: int = 15) -> str:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=timeout,
        )
        return result.stdout
