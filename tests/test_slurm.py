"""Tests for Slurm job management.

Uses mock subprocess calls so tests run without a real Slurm installation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chem_copilot.slurm.manager import SlurmManager, JobInfo


@pytest.fixture
def mgr(tmp_path: Path) -> SlurmManager:
    """Create a SlurmManager with temp directories."""
    return SlurmManager(
        jobs_dir=tmp_path / "jobs",
        scratch_dir="/scratch/test-chem-copilot",
        fhiaims_binary="/home/albd/projects/fhiaims/build/aims.x",
        fhiaims_species_dir="/home/albd/projects/fhiaims/species_defaults/defaults_2020",
    )


SAMPLE_CONTROL = """\
xc                 pbe
spin               none
relativistic       atomic_zora scalar
sc_accuracy_rho    1e-06
occupation_type    gaussian 0.01
"""

SAMPLE_GEOMETRY = """\
atom     0.0000000000     0.0000000000     0.1173000000 O
atom     0.0000000000     0.7572000000    -0.4692000000 H
atom     0.0000000000    -0.7572000000    -0.4692000000 H
"""


class TestExtractElements:
    def test_water(self, mgr: SlurmManager) -> None:
        assert mgr._extract_elements(SAMPLE_GEOMETRY) == ["O", "H"]

    def test_periodic(self, mgr: SlurmManager) -> None:
        geom = "lattice_vector 5.0 0.0 0.0\natom_frac 0.0 0.0 0.0 Si\natom_frac 0.5 0.5 0.5 Si\n"
        assert mgr._extract_elements(geom) == ["Si"]


class TestSpeciesFilePaths:
    def test_returns_correct_paths(self, mgr: SlurmManager) -> None:
        paths = mgr._species_file_paths(["O", "H"], "light")
        assert len(paths) == 2
        assert "08_O_default" in paths[0]
        assert "01_H_default" in paths[1]
        assert "/light/" in paths[0]


class TestSbatchScript:
    def test_fhiaims_script_structure(self, mgr: SlurmManager) -> None:
        from chem_copilot.slurm.manager import SlurmConfig
        cfg = SlurmConfig(partition="cpu", cpus_per_task=8, time="2:00:00")
        species = ["/path/to/08_O_default", "/path/to/01_H_default"]

        script = mgr._fhiaims_full_script(
            "/scratch/test/job1", "h2o_pbe", cfg,
            SAMPLE_CONTROL, SAMPLE_GEOMETRY, species,
        )
        assert "#!/bin/bash" in script
        assert "#SBATCH --partition=cpu" in script
        assert "#SBATCH --ntasks=8" in script  # MPI ranks = cpus_per_task
        assert "mkdir -p /scratch/test/job1" in script
        assert "mpirun -np $SLURM_NTASKS" in script
        assert "ulimit -s unlimited" in script
        assert "aims" in script
        assert "cat control.in.header" in script
        assert "08_O_default" in script

    def test_psi4_script_structure(self, mgr: SlurmManager) -> None:
        from chem_copilot.slurm.manager import SlurmConfig
        cfg = SlurmConfig(partition="cpu", cpus_per_task=16)

        script = mgr._psi4_full_script(
            "/scratch/test/psi4job", "psi4_test", cfg,
            "import psi4\npsi4.energy('hf')\n",
            "/home/albd/projects/activate_xdm.sh",
        )
        assert "#!/bin/bash" in script
        assert "#SBATCH --partition=cpu" in script
        assert "source /home/albd/projects/activate_xdm.sh" in script
        assert "python input.py" in script
        assert "import psi4" in script


class TestSubmitFHIAims:
    @patch("chem_copilot.slurm.manager.subprocess.run")
    def test_creates_local_metadata(
        self, mock_run: MagicMock, mgr: SlurmManager
    ) -> None:
        mock_run.return_value = MagicMock(
            stdout="Submitted batch job 12345\n", returncode=0
        )
        job_id = mgr.submit_fhiaims(
            control_in=SAMPLE_CONTROL,
            geometry_in=SAMPLE_GEOMETRY,
            job_name="h2o_test",
            basis_tier="light",
        )
        assert job_id == "12345"

        local_dir = Path(mgr._find_local_dir("12345"))
        assert local_dir.exists()
        assert (local_dir / ".job_id").read_text().strip() == "12345"
        assert (local_dir / ".program").read_text().strip() == "fhi-aims"
        assert (local_dir / "control.in.header").exists()
        assert (local_dir / "geometry.in").exists()
        assert (local_dir / "run.sh").exists()
        assert (local_dir / ".remote_dir").exists()

    @patch("chem_copilot.slurm.manager.subprocess.run")
    def test_sbatch_called(self, mock_run: MagicMock, mgr: SlurmManager) -> None:
        mock_run.return_value = MagicMock(
            stdout="Submitted batch job 99999\n", returncode=0
        )
        mgr.submit_fhiaims(SAMPLE_CONTROL, SAMPLE_GEOMETRY)
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0][0] == "sbatch"

    @patch("chem_copilot.slurm.manager.subprocess.run")
    def test_run_script_has_mpirun(self, mock_run: MagicMock, mgr: SlurmManager) -> None:
        mock_run.return_value = MagicMock(stdout="Submitted batch job 100\n", returncode=0)
        mgr.submit_fhiaims(SAMPLE_CONTROL, SAMPLE_GEOMETRY)
        local_dir = Path(mgr._find_local_dir("100"))
        script = (local_dir / "run.sh").read_text()
        assert "mpirun" in script
        assert "aims" in script


class TestSubmitPsi4:
    @patch("chem_copilot.slurm.manager.subprocess.run")
    def test_creates_psi4_job(self, mock_run: MagicMock, mgr: SlurmManager) -> None:
        mock_run.return_value = MagicMock(stdout="Submitted batch job 200\n", returncode=0)
        job_id = mgr.submit_psi4(
            input_script="import psi4\npsi4.energy('hf')\n",
            job_name="water_hf",
        )
        assert job_id == "200"
        local_dir = Path(mgr._find_local_dir("200"))
        assert (local_dir / ".program").read_text().strip() == "psi4"
        script = (local_dir / "run.sh").read_text()
        assert "activate_xdm" in script
        assert "python input.py" in script


class TestGetJobStatus:
    @patch("chem_copilot.slurm.manager.subprocess.run")
    def test_running_job(self, mock_run: MagicMock, mgr: SlurmManager) -> None:
        mock_run.return_value = MagicMock(
            stdout="12345|h2o_test|RUNNING|odin|0:05:32\n", returncode=0
        )
        info = mgr.get_job_status("12345")
        assert info.state == "RUNNING"
        assert info.node == "odin"

    @patch("chem_copilot.slurm.manager.subprocess.run")
    def test_completed_via_sacct(self, mock_run: MagicMock, mgr: SlurmManager) -> None:
        def side_effect(cmd, **kwargs):
            if cmd[0] == "squeue":
                return MagicMock(stdout="", returncode=0)
            elif cmd[0] == "sacct":
                return MagicMock(
                    stdout="12345|h2o|COMPLETED|odin|0:02:15|0:0\n", returncode=0
                )
            return MagicMock(stdout="", returncode=0)

        mock_run.side_effect = side_effect
        info = mgr.get_job_status("12345")
        assert info.state == "COMPLETED"


class TestListRecentJobs:
    @patch("chem_copilot.slurm.manager.subprocess.run")
    def test_lists_submitted_jobs(self, mock_run: MagicMock, mgr: SlurmManager) -> None:
        mock_run.return_value = MagicMock(stdout="Submitted batch job 111\n", returncode=0)
        mgr.submit_fhiaims(SAMPLE_CONTROL, SAMPLE_GEOMETRY, job_name="j1")

        mock_run.return_value = MagicMock(stdout="Submitted batch job 222\n", returncode=0)
        mgr.submit_fhiaims(SAMPLE_CONTROL, SAMPLE_GEOMETRY, job_name="j2")

        mock_run.side_effect = lambda cmd, **kw: MagicMock(stdout="", returncode=0)
        jobs = mgr.list_recent_jobs(5)
        assert len(jobs) == 2
        assert {j.job_id for j in jobs} == {"111", "222"}


class TestToolWrappers:
    def test_all_slurm_tools_importable(self) -> None:
        from chem_copilot.tools import (
            submit_fhiaims_job, submit_psi4_job,
            check_job_status, list_recent_jobs, cancel_job, get_job_result,
        )
        assert submit_fhiaims_job.name == "submit_fhiaims_job"
        assert submit_psi4_job.name == "submit_psi4_job"
        assert check_job_status.name == "check_job_status"

    def test_tool_count(self) -> None:
        from chem_copilot.tools import ALL_TOOLS
        assert len(ALL_TOOLS) == 9
        names = {t.name for t in ALL_TOOLS}
        assert "submit_fhiaims_job" in names
        assert "submit_psi4_job" in names
        assert "get_job_result" in names
