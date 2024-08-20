# github-repo-explorer

## What does this do?

Scrapes issues and prs from GitHub and saves them in a parquet file.
It also provides analytics in a streamlit dashboard.

## How do I use this?

Install cruft and uv on your system e.g. in your "base" env
```
pipx install cruft uv
```

Fill in the template like below
```
cruft create git@github.com:raybellwaves/github-repo-explorer.git
github_organization: e.g. dask
github_repository: e.g. dask
llm_chat_framework: e.g. openai
embeddings_framework: e.g. openai
llm_agent_framework: e.g. google (large context)
geolocater_framework: e.g. photon
repository_name: ENTER for default
created_after_date: e.g. 2024-01-01 if just want content after a certain data
current_date: ENTER for default

```

You can then do:
```
cd {{cookiecutter.github_repository}}-repo-explorer && \
mamba create -n gre python=3.11 --y && \
  conda activate gre && \
  uv pip install -r requirements-dev.txt --find-links https://download.pytorch.org/whl/cpu
python main.py run_all --states open closed --content_types issues prs --verbose True
streamlit run main.py
```

See the README in the folder for more information.

## Examples of projects using this template

 - XXX

## TODO

 - Could include timeline API which links issues/PRs

 ## See also

 - https://devlake.apache.org/
 - https://github.com/dlvhdr/gh-dash
