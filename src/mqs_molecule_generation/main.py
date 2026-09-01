"""Main entry point for mqs-molecule-generation.

Simply delegates to the CLI commands module.
"""

from mqs_molecule_generation.cli.commands import main

if __name__ == "__main__":
    main()
