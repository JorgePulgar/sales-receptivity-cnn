---
name: feedback-no-env-hunting
description: Don't hunt for the Python interpreter or conda env path when already inside the project folder
metadata:
  type: feedback
---

Don't search the filesystem for the Python executable or conda environment when already inside the project directory. The conda env (`sales-cnn`) is not active in the bash shell used by tools, so smoke tests will fail with "module not found". Just write correct code and commit — the user validates in their own terminal.

**Why:** Searching with `find` across the user's home directory triggers unnecessary permission prompts and wastes time. The project is already in the correct folder.

**How to apply:** Write the code, commit it, and note that the user should run validation in their conda environment (`conda activate sales-cnn`). Do not attempt to locate or activate the env from within tool calls.
