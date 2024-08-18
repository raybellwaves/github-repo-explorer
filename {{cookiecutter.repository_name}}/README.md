# {{cookiecutter.repository_name}}

## CLI commands

To update your repo with the latest template (if there are updates) run:
```
cruft update
```

Create an environment for development
```
cd {{cookiecutter.github_repository}}-repo-explorer && \
mamba create -n gre python=3.11 --y && \
  conda activate gre && \
  uv pip install -r requirements-dev.txt --find-links https://download.pytorch.org/whl/cpu
```

Remove the environment
```
conda remove --name gre --all --y
```

To scrape all content, summarize the titles using an LLM and geolocate users run
```
python main.py run_all --states open closed --content_types issues prs --verbose True
```

Scraping github:
```
# Scrape just open issues
python main.py scrape_gh --states open --content_types issues --verbose True
# Scrape just closed issues
python main.py scrape_gh --states closed --content_types issues --verbose True
# Scrape just open PRs
python main.py scrape_gh --states open --content_types prs --verbose True
# Scrape just closed PRs
python main.py scrape_gh --states closed --content_types prs --verbose True
# Scrape open and closed issues and open and closed PRs
# for largish REPOs e.g. latest issue number is 10,000, run this over night
python main.py scrape_gh --states open closed --content_types issues prs --verbose True
```

Create DataFrame (Concatenate and flatten files):
```
# just open issues
python main.py create_df --states open --content_types issues
# open and closed issues and open and closed PRs
python main.py create_df --states open closed --content_types issues prs
```

Create a vector database
```
# just open issues
python main.py create_vector_db --states open --content_types issues
# open and closed issues and open and closed PRs
python main.py create_vector_db --states open closed --content_types issues prs
```

Run the dashboard
```
streamlit run main.py
```

Deploy the dashboard
```
...
```
