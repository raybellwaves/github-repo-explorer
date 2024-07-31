# {{cookiecutter.repository_name}}

## CLI commands

To update your repo with the latest template (if there are updates) run:
```
cruft update
```

Create an environment for development
```
mamba create -n gre python=3.11 --y && \
  conda activate gre && \
  uv pip install -r requirements-dev.txt --find-links https://download.pytorch.org/whl/cpu
```

Remove the environment
```
conda remove --name gie --all --y
```

Scraping github:
```
cd {{cookiecutter.github_repository}}-repo-explorer
# Scrape just open issues
python main.py scrape_gh --states open --content_types issues --verbose True
# Scrape just closed issues
python main.py scrape_gh --states closed --content_types issues --verbose True
# Scrape just open PRs
python main.py scrape_gh --states open --content_types prs --verbose True
# Scrape just closed PRs
python main.py scrape_gh --states closed --content_types prs --verbose True
# Scrape open and closed issues and open and closed PRs
python main.py scrape_gh --states open closed --content_types issues prs --verbose True
```

Create DataFrame (Concatenate and flatten files):
```
# Concat just open issues
python main.py create_df --states open --content_types issues
# Concat open and closed issues and open and closed PRs
python main.py create_df --states open closed --content_types issues prs
```