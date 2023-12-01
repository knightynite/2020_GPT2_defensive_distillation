# GPT-2 + defensive distillation — runtime image. PyTorch 1.6 + transformers 3.x.
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/cache

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY README.md ./

RUN mkdir -p /cache && chmod -R a+rw /cache

ENTRYPOINT ["python"]
CMD ["-m", "src.gpt2_generate", "--prompt", "The most important security risk of LLMs is"]
