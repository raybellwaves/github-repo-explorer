# {{cookiecutter.repository_name}}

## CLI commands

To update your repo with the latest template run:
```
cruft update
```

Create an environment for development
```
mamba create -n gie python=3.11 --y && \
  conda activate gie && \
  uv pip install -r requirements-dev.txt --find-links https://download.pytorch.org/whl/cpu
```

Remove the environment
```
conda remove --name gie --all --y
```

Scraping github options:
```
cd {{cookiecutter.github_repository}}-issue-explorer
python main.py --states open --content_types issues --verbose True
python main.py --states closed --content_types issues --verbose True
python main.py --states open --content_types prs --verbose True
python main.py --states closed --content_types prs --verbose True
```

Scrape everything
```
cd {{cookiecutter.github_repository}}-issue-explorer
python main.py --states open --content_types issues --verbose True
```
