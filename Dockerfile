# CPU data preparation / baselines only. The Mac GPU model runs on the host.
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY configs ./configs
ENTRYPOINT ["mami"]
CMD ["inspect"]
