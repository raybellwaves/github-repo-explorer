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

Scraping github:
```
cd {{cookiecutter.github_repository}}-issue-explorer
python main.py scrape_gh --states open --content_types issues --verbose True
python main.py scrape_gh --states closed --content_types issues --verbose True
python main.py scrape_gh --states open --content_types prs --verbose True
python main.py scrape_gh --states closed --content_types prs --verbose True
python main.py scrape_gh --states open closed --content_types issues prs
```

Concating files:
```
python main.py concat_files --states open --content_types issues --verbose True

```