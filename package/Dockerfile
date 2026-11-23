# A clean machine: Debian with no Python, no Node and no git. The researcher's steps only:
# install uv, install the wheel, build the library, start the app (here driven by e2e.py).
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home researcher
USER researcher
WORKDIR /home/researcher
ENV PATH="/home/researcher/.local/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# owned by the researcher, so the data volume mounted here is too
RUN mkdir -p /home/researcher/.local/share/asclepius
COPY --chown=researcher asclepius.whl e2e.py ./
ARG WHEEL
RUN mv asclepius.whl "$WHEEL" && uv tool install --python 3.14 "./$WHEEL"
# a library kept in the data volume from an earlier run is refreshed, not fetched again
CMD ["sh", "-c", "if [ -f .local/share/asclepius/library.sqlite ]; then asclepius-library refresh; else asclepius-library build; fi && asclepius-library index && uv run --no-project python e2e.py"]
