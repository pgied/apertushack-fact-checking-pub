# The image the organisers run: linux/amd64, CPU only, no secrets inside. `make run` builds and runs it:
#
#     docker build --platform linux/amd64 -t apertushack-fact-checking .
#     docker run --rm --platform linux/amd64 -e BASE_URL -e API_KEY \
#         -v "$PWD/data/examples:/data:ro" -v "$PWD/output:/output" \
#         apertushack-fact-checking --input /data/cases.jsonl --output /output/predictions.jsonl
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

WORKDIR /app
# The image's Python, not a downloaded one; bytecode compiled at build time, so that a run starts quickly
ENV UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
# Only what the project itself needs: no torch and no test tools, see the dependency groups in pyproject.toml
RUN uv sync --locked --no-default-groups
COPY conf ./conf
COPY src ./src

ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["python", "-m", "src.entrypoint"]
