"""LangGraph-compatible tool wrappers for computational chemistry."""

from chem_copilot.tools.generate_fhiaims import generate_fhiaims_input
from chem_copilot.tools.generate_psi4 import generate_psi4_input
from chem_copilot.tools.parse_output import parse_calculation_output
from chem_copilot.tools.slurm_tools import (
    cancel_job,
    check_job_status,
    get_job_result,
    list_recent_jobs,
    submit_fhiaims_job,
    submit_psi4_job,
)

ALL_TOOLS = [
    generate_fhiaims_input,
    generate_psi4_input,
    parse_calculation_output,
    submit_fhiaims_job,
    submit_psi4_job,
    check_job_status,
    list_recent_jobs,
    cancel_job,
    get_job_result,
]

__all__ = [
    "generate_fhiaims_input",
    "generate_psi4_input",
    "parse_calculation_output",
    "submit_fhiaims_job",
    "submit_psi4_job",
    "check_job_status",
    "list_recent_jobs",
    "cancel_job",
    "get_job_result",
    "ALL_TOOLS",
]
