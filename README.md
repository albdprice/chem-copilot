# Chem Copilot

An agentic AI system for computational chemistry workflows. Chem Copilot helps researchers set up, submit, monitor, and interpret electronic structure calculations using **FHI-aims**, **Psi4**, and **Gaussian**.

Built with [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration, [ASE](https://wiki.fysik.dtu.dk/ase/) for structure handling, and [cclib](https://cclib.github.io/) for output parsing.

## Features

**Input Generation**
- FHI-aims control.in + geometry.in with full support for XC functionals (PBE, PBE0, HSE06, B86bPBE, SCAN, ...), basis tiers (light through really_tight), dispersion corrections (TS, MBD, XDM with explicit or keyword damping), spin-polarized calculations, periodic systems with k-grids, geometry optimization, and molecular dynamics
- Psi4 Python scripts for HF, DFT, MP2, CCSD(T) with Dunning/Ahlrichs basis sets, D3/D3BJ/D4 dispersion, frozen core, and symmetry handling
- Generated inputs validated against FHI-aims regression test suite

**Job Management**
- Slurm job submission with automatic species defaults assembly
- FHI-aims MPI jobs with correct `ulimit -s unlimited` and `mpirun` configuration
- Psi4 jobs with conda environment activation (psi4-xdm / psi4-apbe0)
- Job monitoring, listing, cancellation, and result retrieval

**Output Parsing**
- Custom FHI-aims parser (cclib lacks FHI-aims support) handling Fortran scientific notation
- Psi4 and Gaussian parsing via cclib with explicit parser fallback
- Extracts: total energies, SCF trajectories, orbital eigenvalues, HOMO-LUMO gaps, forces, vibrational frequencies, optimization convergence
- Structured Pydantic output models

**Agent**
- ReAct-style LangGraph agent with 9 tools
- Supports Ollama (local, tested with qwen2.5-coder:7b) and Claude API backends
- Handles both native tool calls and text-based JSON tool calls (robust across model sizes)
- Conversation memory via LangGraph checkpointer
- Domain-expert system prompt for computational chemistry

**API**
- FastAPI REST endpoint with chat, streaming (SSE), and job management routes
- CORS-enabled for frontend integration

## Architecture

```
User (CLI / API / Chat)
  |
  v
LangGraph Agent (ReAct loop)
  |-- generate_fhiaims_input    Generate FHI-aims control.in + geometry.in
  |-- generate_psi4_input       Generate Psi4 Python input scripts
  |-- submit_fhiaims_job        Submit to Slurm (MPI, species defaults)
  |-- submit_psi4_job           Submit to Slurm (conda env activation)
  |-- check_job_status          Query squeue / sacct
  |-- list_recent_jobs          List submitted jobs
  |-- cancel_job                scancel
  |-- get_job_result            Fetch + parse output from compute node
  |-- parse_calculation_output  Parse any output file directly
```

## Installation

```bash
# Core (input generation + parsing)
pip install -e .

# With agent dependencies (LangGraph + Ollama)
pip install -e .

# With Claude API support
pip install -e ".[claude]"

# With FastAPI server
pip install -e ".[api]"

# Development
pip install -e ".[dev]"
```

## Quick Start

### CLI Agent

```bash
# With Ollama
export OLLAMA_BASE_URL=http://your-ollama-server:11434
python examples/demo_agent.py

# With Claude API
export ANTHROPIC_API_KEY=sk-...
python examples/demo_agent.py --backend claude

# Single query
python examples/demo_agent.py --query "Set up a PBE0/tight FHI-aims calculation for benzene with XDM dispersion"
```

### API Server

```bash
export OLLAMA_BASE_URL=http://your-ollama-server:11434
uvicorn chem_copilot.api.app:app --host 0.0.0.0 --port 8000

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Generate FHI-aims input for a water molecule with PBE/light"}'

# Stream
curl -N http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Set up CCSD(T)/cc-pVTZ for water"}'

# Jobs
curl http://localhost:8000/jobs
curl http://localhost:8000/jobs/12345
```

### Python API

```python
from chem_copilot.generators import FHIAimsGenerator, Psi4Generator
from chem_copilot.models import MoleculeSpec, FHIAimsParams, XCFunctional, BasisSetTier, DispersionMethod, XDMParams

# Define a molecule
h2o = MoleculeSpec(
    symbols=["O", "H", "H"],
    coords=[[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]],
)

# Generate FHI-aims input: PBE0/tight with XDM dispersion
params = FHIAimsParams(
    xc=XCFunctional.PBE0,
    basis=BasisSetTier.TIGHT,
    dispersion=DispersionMethod.XDM,
    xdm_params=XDMParams(keyword="bj tight"),
    compute_forces=True,
)
result = FHIAimsGenerator().generate(h2o, params)
print(result.files["control.in"])

# Parse an existing output file
from chem_copilot.parsers import OutputParser
parsed = OutputParser().parse_file("path/to/aims.out")
print(f"Energy: {parsed.energies.total_energy:.6f} eV")
print(f"HOMO-LUMO gap: {parsed.orbitals.homo_lumo_gap:.4f} eV")
```

## Testing

```bash
# Run all tests (176 passing)
pytest tests/ -v

# With coverage
pytest tests/ --cov=chem_copilot --cov-report=html
```

Tests include:
- **Unit tests**: Pydantic models, generators, parser methods (mock data)
- **Regression tests**: FHI-aims output validated against xdm-diamond, H2O.pbe.relaxation, GMTKN55 conventions
- **Integration tests**: Real output files from FHI-aims (PBE0 He, butane, H2O relaxation), Psi4 (DFT, CCSD(T), failed SCF), Gaussian (PBE0/aug-cc-pVTZ)
- **Agent tests**: End-to-end tool execution with fake LLMs, conversation memory
- **Slurm tests**: Job submission, monitoring, result retrieval (mocked subprocess)

## Project Structure

```
src/chem_copilot/
  models.py           Pydantic models for molecules, parameters, results
  generators/
    fhi_aims.py       FHI-aims control.in + geometry.in generator
    psi4.py           Psi4 Python input script generator
  parsers/
    output_parser.py  FHI-aims (custom) + Psi4/Gaussian (cclib) parser
  tools/              LangGraph @tool wrappers for all operations
  agent/
    graph.py          Custom LangGraph ReAct agent with text-based tool call parsing
    prompt.py         Domain-expert system prompt
  slurm/
    manager.py        Slurm job submission, monitoring, result retrieval
  llm/
    config.py         Ollama / Claude API backend configuration
  api/
    app.py            FastAPI REST + SSE streaming endpoint
tests/                176 tests across 9 test modules
examples/             CLI demo agent
```

## Supported Methods

| Code | Methods | Basis Sets | Dispersion |
|------|---------|------------|------------|
| FHI-aims | LDA, PBE, PBE0, HSE06, B86bPBE, SCAN, TPSS, B3LYP, M06-2X | light, intermediate, tight, really_tight | TS, MBD, MBD-NL, XDM |
| Psi4 | HF, DFT (any functional), MP2, MP3, CCSD, CCSD(T), SAPT0 | Dunning (cc-pVXZ, aug-), Ahlrichs (def2-) | D3, D3BJ, D4 |
| Gaussian | (parsing only) | (parsing only) | (parsing only) |

## License

MIT
