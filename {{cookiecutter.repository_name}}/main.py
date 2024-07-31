import os

import pandas as pd

ORG = "{{cookiecutter.github_organization}}"
REPO = "{{cookiecutter.github_repository}}"
LLM_FRAMEWORK = "{{cookiecutter.llm_framework}}"
GEOLOCATER_FRAMEWORK = "{{cookiecutter.geolocater_framework}}"
SNAPSHOT_FOLDER = "snapshot_{{ cookiecutter.current_date }}"
CREATED_AFTER_DATE = pd.Timestamp("{{cookiecutter.created_after_date}}", tz="UTC")

BOTS = [
    "GPUtester",
    "codecov[bot]",
    "dependabot[bot]",
    "github-actions[bot]",
    "pep8speaks",
    "pre-commit-ci[bot]",
    "review-notebook-app[bot]",
]

ISSUE_PR_COLUMNS = [
    "author_association",
    "body",
    "comments",
    "created_at",
    "label_names",
    "number",
    "reactions.+1",
    "reactions.-1",
    "reactions.confused",
    "reactions.eyes",
    "reactions.heart",
    "reactions.hooray",
    "reactions.laugh",
    "reactions.rocket",
    "reactions.total_count",
    "state",
    "title",
    "url",
    "user.login",
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


def scrape_gh(
    states: list[str] = ["open", "closed"],
    content_types: list[str] = ["issues", "prs"],
    verbose: bool = False,
) -> None:
    """
    Puts data into eight folders:
    open_issues, closed_issues, open_prs, closed_prs
    These contain the titles and body

    open_issues_comments, closed_issues_comments, open_prs_comments, closed_prs_comments
    These contain the comments.

    GitHub shares the same structure for issues and PRs
    Note: not tested for only open issues for example
    """
    if verbose:
        print(f"{states=}, {content_types=}")
    import requests
    import json
    import pandas as pd

    from tqdm.auto import tqdm

    GH_API_URL_PREFIX = f"https://api.github.com/repos/{ORG}/{REPO}/"
    headers = {"Authorization": f"token {GITHUB_API_TOKEN}"}

    users = set()
    for state in states:
        if verbose:
            print(f"{state=}")
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
            # keep issues or prs after CREATED_AFTER_DATE
            page_issues_or_prs = [
                page_issue_or_pr
                for page_issue_or_pr in page_issues_or_prs
                if pd.Timestamp(page_issue_or_pr["created_at"]) >= CREATED_AFTER_DATE
            ]

            for content_type in content_types:
                if verbose:
                    print(f"{content_type=}")
                folder = f"{SNAPSHOT_FOLDER}/{state}_{content_type}"
                os.makedirs(folder, exist_ok=True)
                comment_folder = f"{SNAPSHOT_FOLDER}/{state}_{content_type}_comments"
                os.makedirs(comment_folder, exist_ok=True)
                if content_type == "issues":
                    endpoint = "issues"
                    page_issues_or_prs_filtered = [
                        issue
                        for issue in page_issues_or_prs
                        if "pull_request" not in issue
                    ]
                elif content_type == "prs":
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
                    f"fetching {state} {content_type} for page {page}",
                ):
                    number = issue_or_pr["number"]
                    padded_number = f"{number:06d}"
                    filename = (
                        f"{folder}/{content_type[:-1]}_detail_{padded_number}.json"
                    )
                    if os.path.exists(filename):
                        if verbose:
                            print(f"{filename} already exists")
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

                        # Reactions for PRs can be found in the issue endpoint
                        if content_type == "prs":
                            detail_url2 = f"{GH_API_URL_PREFIX}issues/{number}"
                            if verbose:
                                print(f"{detail_url2=}")
                            detail_response2 = requests.get(
                                detail_url2, headers=headers, timeout=10
                            )
                            if not _status_code_checks(detail_response2.status_code):
                                break
                            detail_response2_json = detail_response2.json()
                            if not _json_content_check(detail_response2_json):
                                break
                            detail_response_json["reactions"] = detail_response2_json[
                                "reactions"
                            ]
                        with open(filename, "w") as f:
                            json.dump(detail_response_json, f, indent=4)
                        users.add(detail_response_json["user"]["login"])
                        # Reactions for PRs can be found in the issue endpo
                        # Get comments data
                        if detail_response_json["comments"] > 0:
                            filename = (
                                f"{comment_folder}/" f"comments_{padded_number}.json"
                            )
                            if os.path.exists(filename):
                                if verbose:
                                    print(f"{filename} already exists")
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
                                users.add(detail_response_json["user"]["login"])
            page += 1
    # Scrape users
    users_list = list(users)
    folder = f"{SNAPSHOT_FOLDER}/users"
    os.makedirs(folder, exist_ok=True)
    for username in tqdm(users_list, "fetching data for users"):
        user_detail_response = requests.get(
            f"https://api.github.com/users/{username}",
            headers=headers,
        )
        if not _status_code_checks(user_detail_response.status_code):
            break
        user_detail = user_detail_response.json()
        if not _json_content_check(detail_response2_json):
            break
        # Add geo column
        if GEOLOCATER_FRAMEWORK != "None":
            if GEOLOCATER_FRAMEWORK == "photon":
                from geopy.geocoders import Photon

                geolocator = Photon()
                geocoded_location = geolocator.geocode(user_detail["location"])
                if geocoded_location is not None:
                    user_detail["location_lat"] = geocoded_location.latitude
                    user_detail["location_lon"] = geocoded_location.longitude
                else:
                    user_detail["location_lat"] = None
                    user_detail["location_lon"] = None
        file_path = os.path.join(folder, f"user_detail_{username}.json")
        with open(file_path, "w") as f:
            json.dump(user_detail, f, indent=4)
    return None


def create_df(
    states: list[str] = ["open", "closed"],
    content_types: list[str] = ["issues", "prs"],
) -> None:
    """
    Concat data into a dataframe.

    Row UUID is content_type number + comment
    """
    from glob import glob
    import json
    import requests
    import os
    import pandas as pd

    from tqdm.auto import tqdm

    for state in states:
        for content_type in content_types:
            folder = f"{SNAPSHOT_FOLDER}/{state}_{content_type}"
            files = sorted(os.listdir(folder))
            df = pd.DataFrame()
            for file in tqdm(files, f"concatenating {state} {content_type}"):
                padded_number = file.split("_")[-1].split(".")[0]
                with open(f"{folder}/{file}", "r") as f:
                    data = json.load(f)
                _df = pd.json_normalize(data)
                if _df["body"][0] is None:
                    _df["body"] = ""
                _df["label_names"] = _df["labels"].apply(
                    lambda x: [label["name"] for label in x]
                    if isinstance(x, list)
                    else []
                )
                _df["created_at"] = pd.Timestamp(_df["created_at"][0])
                _df["url"] = _df["url"][0].replace(
                    "https://api.github.com/repos/", "https://github.com/"
                )
                _df["user.url"] = _df["user.url"][0].replace(
                    "https://api.github.com/users/", "https://github.com/"
                )
                if LLM_FRAMEWORK == "openai":
                    _df["LLM_title_subject"] = _chat_response(
                        "Give me a one word summary of the following GitHub "
                        f"{REPO} {content_type[:-1]} title: {_df['title'][0]}"
                    )
                # Keep useful columns
                if LLM_FRAMEWORK != "None":
                    _df = _df[ISSUE_PR_COLUMNS + "LLM_title_subject"]
                else:
                    _df = _df[ISSUE_PR_COLUMNS]
                _df = _df.rename({"comments": "n_comments"})

                # Read comment data if exists
                comment_file = (
                    f"{SNAPSHOT_FOLDER}/{state}_{content_type}_comments/"
                    f"comments_{padded_number}.json"
                )
                if os.path.exists(comment_file):
                    with open(comment_file, "r") as f:
                        data = json.load(f)
                    _df2 = pd.json_normalize(data)
                    _df2["created_at"] = pd.to_datetime(_df2["created_at"])
                    _df2["user.url"] = _df2["user.url"].str.replace(
                        "https://api.github.com/users/", "https://github.com/"
                    )
                    _df2["number"] = int(padded_number)
                    _df2 = _df2[["body", "created_at", "number", "user.login"]]
                    _df2 = _df2.rename(
                        columns={
                            "body": "comment",
                            "created_at": "comment_created_at",
                            "user.login": "commenter",
                        }
                    )
                else:
                    # empty comment df
                    _df2 = pd.DataFrame(
                        columns=["comment", "comment_created_at", "number", "commenter"]
                    )
                    _df2["number"] = int(padded_number)
                _df = pd.merge(_df, _df2, on="number", how="left")
                df = pd.concat([df, _df], axis=0).reset_index(drop=True)
                file = f"{SNAPSHOT_FOLDER}/{state}_{content_type}_dataframe.csv"
                df.to_csv(file, index=False)
    files = glob(f"{SNAPSHOT_FOLDER}/*.csv")
    users = set()
    for f in files:
        df = pd.read_csv(f)
        users.update(df["user.login"].unique())
    users_list = list(users)
    folder = f"{SNAPSHOT_FOLDER}/users"
    os.makedirs(folder, exist_ok=True)
    user_df = pd.DataFrame()
    for username in tqdm(users_list, "fetching data for users"):
        user_detail_response = requests.get(
            f"https://api.github.com/users/{username}",
            headers={"Authorization": f"token {os.environ['GITHUB_API_TOKEN']}"},
        )
        user_detail = user_detail_response.json()
        # Add geo column
        file_path = os.path.join(folder, f"user_detail_{username}.json")
        with open(file_path, "w") as f:
            json.dump(user_detail, f, indent=4)
        user_df = pd.concat([user_df, pd.json_normalize(user_detail)], axis=0)

    return None


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
    elif args.function == "create_df":
        create_df(states=args.states, content_types=args.content_types)
    else:
        print(f"Unknown function: {args.function}")
