# TODO (Seed)

Seed development tasks to construct the project.
Nothing here shall be edited after initial construction.

## Instruction

```
create a high quality engine that is capable of performing inference very efficiently for the Liquid language architecture.

It should use pre-trained weights provided in the huggingface transformer format:
- https://huggingface.co/LiquidAI/LFM2.5-2.6B

The implementation should be written in pure C, cross-platform, and without dependencies.

The engine should be modular when appropriate and all structures, properties, functions, and parameters should use balanced/rhythmic naming conventions.

Design in a way that will support the possibility of using an accelerator in the future.

All documentation should be logically organized, use clear and direct language, and balanced/rhythmic naming conventions.

The project structure should be flat and restricted to the following files:
- app_core.c: engine library with clean API
- app_main.c: CLI entry point
- app_test.c: unit tests
- app_test.py: comparison to the reference implementation
- run.py: ci/cd workflows: install, build, test, run 
- TODO.md: flat list of open development tasks
- CHANGES.md: development progress and rationale
- GUIDE.md: a complete tour of the project implementation
- README.md: summary and quickstart

Ask me any questions for clarification if needed.
```