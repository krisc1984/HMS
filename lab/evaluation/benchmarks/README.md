# LongMemEval Benchmark Code

This directory contains the LongMemEval runner used by the memory QA
reproduction workflow.

The runner supports two experiment modes through the project-level script:

```text
HMS_PIPELINE=ledger
HMS_PIPELINE=self_evolution
```

Run from the repository root:

```bash
bash .aaaSCRIPT/run_benchmark.sh
```

Retrieval-only mode is enabled by default in `.aaaSCRIPT/run_benchmark.sh`.
Under this setting, the runner reuses existing database memories and skips
ingestion unless explicitly configured otherwise.
