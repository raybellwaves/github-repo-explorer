# github-repo-explorer

## What does this do?

Scrapes issues and prs from GitHub and saves them in a parquet file.

## How do I use this?

Install cruft and uv on your system e.g. in your "base" env
```
pipx install cruft uv
```

Fill in the template like below
```
cruft create git@github.com:raybellwaves/github-issue-explorer.git
dask, dask, openai, ENTER
```

From there follow the README in the new folder you just created.

## Examples of projects using this template

 - XXX

## TODO

 - Could include timeline API which links issues/PRs
 - Haven't done anything with PR data

 ## See also

 - https://devlake.apache.org/
 - https://github.com/dlvhdr/gh-dash
