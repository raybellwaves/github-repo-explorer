import os
import sys

ORG = "{{cookiecutter.github_organization}}"
REPO = "{{cookiecutter.github_repository}}"
LLM_FRAMEWORK = "{{cookiecutter.llm_framework}}"
SNAPSHOT_FOLDER = "snapshot_{{cookiecutter.snapshot_folder}}"

BOTS = [
    "GPUtester",
    "codecov[bot]",
    "dependabot[bot]",
    "github-actions[bot]",
    "pep8speaks",
    "pre-commit-ci[bot]",
    "review-notebook-app[bot]",
]

try:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
except KeyError:
    print("env var OPENAI_API_KEY not found")
    OPENAI_API_KEY = ""
try:
    GITHUB_API_TOKEN = os.environ.get("GITHUB_API_TOKEN")
except KeyError:
    print("env var GITHUB_API_TOKEN not found")
    GITHUB_API_TOKEN = ""


def _status_code_checks(status_code: int) -> bool:
    if status_code == 200:
        return True
    elif status_code == 403 or status_code == 429:
        print("hit rate limit, wait one hour. breaking")
        return False
    else:
        print(f"status code: {status_code}. breaking")
        return False


def _json_content_check(json_content) -> bool:
    if not json_content:
        print("no content found in response")
        return False
    else:
        return True


def _chat_response(content):
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content


def _num_tokens_from_string(
    string: str,
    encoding_name: str = "cl100k_base",
) -> int:
    """Returns the number of tokens in a text string."""
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def _agent_response(agent, content):
    return agent.invoke(content)["output"]


# Core functions
def scrape_gh(
    org: str = ORG,
    repo: str = REPO,
    states: list[str] = ["open", "closed"],
    content_types: list[str] = ["issues", "prs"],
    verbose: bool = False,
) -> None:
    """
    Puts data into 4 folders:
    open_issues, closed_issues, open_prs, closed_prs
    GitHub shares the same structure for issues and PRs
    Note: not tested for only open issues for example
    """
    import requests
    import json

    from tqdm.auto import tqdm

    GH_API_URL_PREFIX = f"https://api.github.com/repos/{org}/{repo}/"
    headers = {"Authorization": f"token {GITHUB_API_TOKEN}"}

    for state in states:
        page = 1
        while True:
            # the issues endpoint is misnomer and contains issues and prs.
            # This returns a high level overview of the issue or pr such as:
            # the user, the body, body reactions e.g.
            # +1 and whether it's a pr or issue
            gh_api_url_suffix = f"issues?state={state}&per_page=100&page={page}"
            if verbose:
                print(f"{gh_api_url_suffix=}")
            url = f"{GH_API_URL_PREFIX}{gh_api_url_suffix}"
            response = requests.get(url, headers=headers)
            if not _status_code_checks(response.status_code):
                break
            # list of ~100 issues or prs from most recent to oldest
            page_issues_or_prs = response.json()
            if not _json_content_check(page_issues_or_prs):
                break
            # Exlude bots
            page_issues_or_prs = [
                page_issue_or_pr
                for page_issue_or_pr in page_issues_or_prs
                if page_issue_or_pr["user"]["login"] not in BOTS
            ]

            for content_type in content_types:
                folder = f"{SNAPSHOT_FOLDER}/{state}_{content_type}"
                os.makedirs(folder, exist_ok=True)
                if content_type == "issues":
                    endpoint = "issues"
                    page_issues_or_prs_filtered = [
                        issue
                        for issue in page_issues_or_prs
                        if "pull_request" not in issue
                    ]
                elif content_type == "prs":
                    pr_comment_folder = f"{SNAPSHOT_FOLDER}/{state}_prs_comments"
                    os.makedirs(pr_comment_folder, exist_ok=True)
                    endpoint = "pulls"
                    page_issues_or_prs_filtered = [
                        pr for pr in page_issues_or_prs if "pull_request" in pr
                    ]
                else:
                    raise ValueError(
                        f"Unknown content type: {content_type}. "
                        "Should be 'issues' or 'prs'"
                    )

                for issue_or_pr in tqdm(
                    page_issues_or_prs_filtered,
                    f"fetching {state} {content_type}",
                ):
                    number = issue_or_pr["number"]
                    padded_number = f"{number:06d}"
                    filename = (
                        f"{folder}/{content_type[:-1]}_detail_{padded_number}.json"
                    )
                    if os.path.exists(filename):
                        continue
                    else:
                        detail_url = f"{GH_API_URL_PREFIX}{endpoint}/{number}"
                        if verbose:
                            print(f"{detail_url=}")
                        detail_response = requests.get(
                            detail_url, headers=headers, timeout=10
                        )
                        if not _status_code_checks(detail_response.status_code):
                            break
                        detail_response_json = detail_response.json()
                        if not _json_content_check(detail_response_json):
                            break
                        # There is also a timeline API that could be included.
                        # This contains information on cross posting issues or prs
                        with open(filename, "w") as f:
                            json.dump(detail_response_json, f, indent=4)
                        if content_type == "prs":
                            # Grab the PR comments as they are in the issue endpoint
                            if detail_response_json["comments"] > 0:
                                filename = (
                                    f"{pr_comment_folder}/"
                                    f"comments_{padded_number}.json"
                                )
                                if os.path.exists(filename):
                                    continue
                                else:
                                    comments_url = (
                                        f"{GH_API_URL_PREFIX}issues/{number}/comments"
                                    )
                                    if verbose:
                                        print(f"{comments_url=}")
                                    comments_response = requests.get(
                                        comments_url, headers=headers, timeout=10
                                    )
                                    if not _status_code_checks(
                                        comments_response.status_code
                                    ):
                                        break
                                    comments_response_json = comments_response.json()
                                    if not _json_content_check(comments_response_json):
                                        break

                                    with open(filename, "w") as f:
                                        json.dump(comments_response_json, f, indent=4)
            page += 1
    return None


def hello_world():
    print("yo")


def concat_files(
    repo: str = REPO,
    states: list[str] = ["open", "closed"],
    content_types: list[str] = ["issues", "prs"],
    llm_cols: bool = True,
    verbose: bool = False,
) -> None:
    from datetime import date
    import json
    import os
    import pandas as pd

    from tqdm.auto import tqdm

    for state in states:
        for content_type in content_types:
            folder = f"snapshot_{date.today()}/{repo}_{state}_{content_type}"
            files = os.listdir(folder)
            df = pd.DataFrame()
            for file in tqdm(files):
                with open(file, "r") as f:
                    data = json.load(f)
                _df = pd.json_normalize(data)
                if _df["body"][0] is None:
                    _df["body"] = ""
                _df["label_names"] = _df["labels"].apply(
                    lambda x: [label["name"] for label in x]
                    if isinstance(x, list)
                    else []
                )
                if llm_cols:
                    _df["LLM_title_subject"] = chat_response(
                        "Give me a one word summary of the following GitHub "
                        f"{repo} {content_type[:-1]} title: {_df['title'][0]}"
                    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("function", type=str, help="Function to call")
    parser.add_argument("--states", nargs="+", type=str, default=["open", "closed"])
    parser.add_argument(
        "--content_types", nargs="+", type=str, default=["issues", "prs"]
    )
    parser.add_argument("--verbose", type=str, default=False)
    args = parser.parse_args()

    if args.function == "scrape_gh":
        scrape_gh(
            states=args.states, content_types=args.content_types, verbose=args.verbose
        )
    elif args.function == "hello_world":
        hello_world()
    else:
        print(f"Unknown function: {args.function}")
