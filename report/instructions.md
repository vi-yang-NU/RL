<!--
The report's "Instructions for running the code" section, which the handout requires. `python run.py report`
puts it into report/report.md and report/report.pdf, after the submission details and before the table of
contents. Keep it short: the table of contents and the dashboard do the navigating.
-->
Nothing has to be run to grade: every answer, plot, evaluation statistic and validation result is in this
report, and the repository includes the files it was built from (`results/`). The dashboard shows the same
answers next to their plots and test evidence, and reruns any part. To open it, with Python 3.9 or newer on
macOS or Linux (WSL on Windows):

```bash
git clone https://github.com/vi-yang-NU/RL.git && cd RL
./setup.sh --no-ssh        # creates .venv and installs the packages (requirements*.txt)
source .venv/bin/activate
python run.py ui
```

It opens at <http://localhost:8501> (if no browser opens, open that address yourself), with one tab per part
of the handout and, in each, a **Run** panel that reruns that part. The rest of this report follows the
handout's order; the table of contents below lists every section. The same parts run from the terminal
(`python run.py -h` lists the commands; `python run.py tests --section dp`, `pg1` or `pg2` reruns the
validation experiments), and the repository's README covers running on deepdish.
