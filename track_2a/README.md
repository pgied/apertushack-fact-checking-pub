# Academia Challenges

> **This project** is our entry for the OST challenge, _Multilingual Natural Language Inference over Swiss Official Voting Booklets_. It checks claims about Swiss popular votes against the official voting booklets with Apertus: does the booklet support the claim (entailment), contradict it (contradiction) or neither (neutral)? It also shows the passage of the booklet that the answer is based on. What the challenge asks for, and how it is judged: the [challenge page](https://hackapertus.notion.site/3deb4fec112a80258fd2ddc61616011d) and its [Solution API Guide](https://hackapertus.notion.site/ef5b4fec112a834da43101ede56300f5).

Submissions must use the Apertus model family.
For Track 2 this means that submitted solutions must be built with Apertus. Other open-weights models can be used to support development, e.g. as automatic judges during evaluation. Their role must be clearly described in the submission report.

💬 In case you have questions, join the conversation on Discord or send an email to “hello@hackapertus.ch”

## How it works
Pick from 5 academia challenges provided by Swiss academic institutions:

- **FHGR:** AI-Powered Job Interview Coach
- **OpenParlData:** Extracting Parliamentary Affairs from PDFs into One Common Structure
- **OST:** Multilingual Natural Language Inference over Swiss Official Voting Booklets
- **UZH:** Detecting Cross-Lingual Semantic Differences in Swiss Government Websites
- **ZHAW:** See It, Say It, Pick It: Vision-Language Grounding for a Real Robot Arm

The challenges incl. submission and judging criteria are described in our **Getting Started guide**:
https://hackapertus.notion.site/getting-started-guide-onlinehack

## Run it

Keep `track_2a/` as it is: don't rename it or move its files, just delete the
other track directories.

From the root of the project:

```bash
make run
```

Fill in the [Makefile](Makefile) so that it works on a clean checkout. It is
expected to run the project in a Docker container, since that is how the judges
will run it, without relying on anything already installed on your machine.

Requirements: Docker. An Apertus v1.5 endpoint, with its key in `API_KEY` and its URL in `BASE_URL`, in the environment or in `.env` (see [`.env.example`](.env.example)); without `BASE_URL` it is the CSCS Inference API. No GPU, no model weights.

`make run` builds the Docker image (linux/amd64, CPU only) and decides the four sample cases in [`data/examples/`](data/examples/). They are the advanced and the beginner case of two dataset rows, checked against one booklet. The predictions go to `output/predictions.jsonl`. `make run DATA=<folder>` decides the `cases.jsonl` of another folder instead, with its booklets in `<folder>/booklets/`.

## The CLI and the Docker image

The image runs the challenge's CLI, [`src/entrypoint.py`](src/entrypoint.py), the way the challenge's Solution API Guide asks:

```bash
docker run --rm --platform linux/amd64 -e BASE_URL -e API_KEY \
    -v "$PWD/data/examples:/data:ro" -v "$PWD/output:/output" \
    ghcr.io/pgied/apertushack-fact-checking-pub:latest \
    --input /data/cases.jsonl --output /output/predictions.jsonl
```

- **Input:** JSONL, one case per line. A case has either a `booklet` (task A, our advanced task) or a `reference` (task B, our beginner task), and a file can mix both. Booklet paths are relative to `/data`.
- **Output:** one line per case, with the label, the evidence and the metrics, and exit code 0. A case that fails is answered neutral, so that no line is missing.
- **Model and strategy:** the ones we submit for each task, `SUBMITTED` in [`src/entrypoint.py`](src/entrypoint.py): Apertus v1.5 70B with [`gied_beginner_scoped`](src/strategies/gied_beginner_scoped/strategy.py) for the beginner task and [`gied_strict_reasoned`](src/strategies/gied_strict_reasoned/strategy.py) for the advanced task. Hydra overrides after the arguments change them for both tasks. Only Apertus v1.5 is called, always through `BASE_URL`.
- **The published image:** `ghcr.io/pgied/apertushack-fact-checking-pub:latest`, built for linux/amd64 by the GitHub workflow [`image.yml`](../.github/workflows/image.yml) on every push to `main`. The submission form has no field for it, so it is named here and in the form's notes.

## Developing

Everything runs from `track_2a/`, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                                    # the environment, with PyTorch and the test tools
uv run python -m src stage=prepare         # the dataset, into data/beginner/ and data/advanced/
uv run python -m src limit=30              # 30 cases with the default strategy, scored, in data/runs/
uv run python -m src.entrypoint --input data/examples/cases.jsonl --output output/predictions.jsonl
uv run pytest                              # the tests
```

### Layout

- [`src/`](src/): the source code. [`src/entrypoint.py`](src/entrypoint.py) is the challenge's CLI. For developing, run `uv run python -m src --help`. The strategies, one folder each, are in [`src/strategies/`](src/strategies/).
- [`conf/`](conf/): the configuration (Hydra): which model, which backend, the generation settings.
- [`tests/`](tests/): the tests, run with `uv run pytest`.
- `data/`: everything generated, not in git: the dataset and one folder per run in `data/runs/`. Only the sample for `make run`, [`data/examples/`](data/examples/), is in git.
- [`docs/`](docs/): the two diagrams of the technical report, as LaTeX (TikZ), with their PDF and SVG; each `.tex` file says how to build them. Our internal notes are kept in a private repository.
- [`slurm/`](slurm/): job scripts for the CSCS cluster.
- [`Dockerfile`](Dockerfile) and [`Makefile`](Makefile): the image, and `make run`.
- [`technical_report.md`](technical_report.md): the technical report, handed in as a PDF.

## Data
The `data/` directory must not exceed 100 MB.

Only [`data/examples/`](data/examples/) is in git, less than 1 MB; `uv run python -m src stage=prepare` creates the rest.


## 📦 Submission Requirements & Deliverables
❗️ Submissions are not handled on Devpost. Submit through our website only:
http://hackapertus.ch/online-hack/submissions

Requirements differ by challenge. See the description of the challenge you are entering for the exact deliverables.

For the OST challenge, the form asks for this repository, the technical report as a PDF of at most 6 pages that includes the token usage and the inference time, and the Apertus model used; no demo video. The organisers pull the published image named above.


## ⚖️ Judging Criteria
Judging criteria also differ by challenge. See the respective challenge description.

For the OST challenge: the macro-F1 of each task, which has to reach 0.60 in task A and 0.70 in task B; the evidence in task A; and the tokens and the processing time per case, relative to the best team. See the [challenge page](https://hackapertus.notion.site/3deb4fec112a80258fd2ddc61616011d).


## Support

**Licensing requirements**
Please check our Terms & Conditions (6. What you build is open source):
https://hackapertus.ch/terms-and-conditions

## FAQ
💡 https://hackapertus.ch/faq

## Contact
💬 In case you have questions, join the conversation on Discord or send an email to “hello@hackapertus.ch”
