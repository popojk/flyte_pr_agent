import subprocess
import tempfile
import logging
import re
import os
import asyncio
from pathlib import Path

import flyte

from flyte.io import Dir, File

# Configure logging at module level for Flyte runtime
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

env = flyte.TaskEnvironment(
    name = "flyte_pr_workflow",
    resources=flyte.Resources(
        memory=("2Gi", "12Gi"),
    ),
    image = flyte.Image.from_debian_base().with_env_vars({"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"), "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN")}).with_apt_packages("git", "make").with_uv_project(pyproject_file=Path("pyproject.toml"), project_install_mode="dependencies_only")
)

@env.task
async def pull_pr_branch(git_url: str) -> Dir:
    """
    Pull PR branch code to local directory

    Args:
        git_url: Git URL, format can be:
                - "https://github.com/owner/repo/pull/123" (GitHub PR URL)
                - "https://github.com/user/repo.git" (regular git URL)

    Returns:
        Dir object containing the cloned repository
    """
    # Get GitHub token from environment variable
    github_token = os.getenv("GITHUB_TOKEN", "")
    # Check if this is a GitHub PR URL
    pr_pattern = r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)'
    pr_match = re.match(pr_pattern, git_url)

    # Create temporary directory for cloned code
    temp_dir = tempfile.mkdtemp(prefix="pr_agent_")
    local_path = Path(temp_dir) / "repo"

    try:
        if pr_match:
            # Extract owner, repo, and PR number
            owner = pr_match.group(1)
            repo = pr_match.group(2)
            pr_number = pr_match.group(3)

            # Build repo URL with token if provided
            if github_token:
                repo_url = f"https://{github_token}@github.com/{owner}/{repo}.git"
            else:
                repo_url = f"https://github.com/{owner}/{repo}.git"

            logging.debug(f"Detected GitHub PR: {owner}/{repo}#{pr_number}")

            # Clone repository
            clone_result = subprocess.run(
                ["git", "clone", repo_url, str(local_path)],
                capture_output=True,
                text=True,
                check=True
            )
            logging.debug(f"Cloned repository successfully")

            # Fetch the PR branch
            fetch_result = subprocess.run(
                ["git", "fetch", "origin", f"pull/{pr_number}/head:pr-{pr_number}"],
                cwd=str(local_path),
                capture_output=True,
                text=True,
                check=True
            )
            logging.debug(f"Fetched PR branch successfully")

            # Checkout the PR branch
            checkout_result = subprocess.run(
                ["git", "checkout", f"pr-{pr_number}"],
                cwd=str(local_path),
                capture_output=True,
                text=True,
                check=True
            )
            logging.debug(f"Checked out PR branch successfully")

        else:
            # Regular git URL - add token if provided
            if github_token and "github.com" in git_url:
                # Insert token into URL
                authenticated_url = git_url.replace("https://github.com", f"https://{github_token}@github.com")
            else:
                authenticated_url = git_url

            # Clone repository
            clone_result = subprocess.run(
                ["git", "clone", authenticated_url, str(local_path)],
                capture_output=True,
                text=True,
                check=True
            )
            logging.debug(f"Cloned repository successfully")

        logging.debug(f"Successfully prepared repository at {local_path}")

        files_dir = await Dir.from_local(local_path)
        return files_dir

    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to clone/checkout repository: {e.stderr}"
        logging.debug(error_msg)
        raise RuntimeError(error_msg)

@env.task
async def flyte_sdk_unit_test(file_dir: Dir):
    """
    Run unit tests using make unit_test command

    Args:
        file_dir: Directory containing the repository code

    Returns:
        Dir object containing the repository
    """
    # Download Dir to local path to execute make command
    local_path = await file_dir.download()
    logging.debug(f"Running unit tests in: {local_path}")

    # Execute make unit_test in the repository directory
    result = subprocess.run(
        ["make", "unit_test"],
        cwd=str(local_path),
        capture_output=True,
        text=True
    )

    # Log test output
    logging.debug("=" * 80)
    logging.debug("Unit test output:")
    logging.debug("=" * 80)
    if result.stdout:
        logging.debug(result.stdout)
    if result.stderr:
        logging.debug("Unit test stderr:")
        logging.debug(result.stderr)

    # Check test result and log accordingly
    if result.returncode == 0:
        logging.debug("=" * 80)
        logging.debug("✓ Unit tests passed successfully!")
        logging.debug("=" * 80)
    else:
        logging.debug("=" * 80)
        logging.debug(f"✗ Unit tests failed with exit code {result.returncode}")
        logging.debug("=" * 80)

@env.task
async def flyte_sdk_lint_fix(file_dir: Dir, git_url: str) -> Dir:
    """
    Run make fmt to format code, commit and push if there are changes

    Args:
        file_dir: Directory containing the repository code
        git_url: GitHub PR URL for extracting owner/repo info

    Returns:
        Dir object containing the formatted repository
    """
    # Download Dir to local path to execute make command
    local_path = await file_dir.download()
    logging.debug(f"Running code formatter in: {local_path}")

    # Execute make fmt in the repository directory
    result = subprocess.run(
        ["make", "fmt"],
        cwd=str(local_path),
        capture_output=True,
        text=True
    )

    # Check if there are any changes after formatting
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(local_path),
        capture_output=True,
        text=True,
        check=True
    )

    changes = status_result.stdout.strip()

    if changes:
        logging.debug(f"Found formatting changes:\n{changes}")

        # Stage all changes
        subprocess.run(
            ["git", "add", "."],
            cwd=str(local_path),
            capture_output=True,
            text=True,
            check=True
        )
        logging.debug("Staged all changes")

        # Commit with signoff
        commit_result = subprocess.run(
            ["git", "commit", "-s", "-m", "auto lint fix"],
            cwd=str(local_path),
            capture_output=True,
            text=True,
            check=True
        )
        logging.debug("=" * 80)
        logging.debug("Commit output:")
        logging.debug(commit_result.stdout)
        logging.debug("=" * 80)

        # Extract owner and repo from git_url
        pr_pattern = r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)'
        pr_match = re.match(pr_pattern, git_url)

        if pr_match:
            owner = pr_match.group(1)
            repo = pr_match.group(2)

            # Get GitHub token from environment
            github_token = os.getenv("GITHUB_TOKEN", "")

            if github_token:
                # Get current branch name
                current_branch_result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    cwd=str(local_path),
                    capture_output=True,
                    text=True,
                    check=True
                )
                current_branch = current_branch_result.stdout.strip()

                logging.debug(f"Pushing to origin/{current_branch}...")

                # Configure git to use token for authentication
                push_url = f"https://{github_token}@github.com/{owner}/{repo}.git"

                # Set remote URL temporarily for this push
                subprocess.run(
                    ["git", "remote", "set-url", "origin", push_url],
                    cwd=str(local_path),
                    capture_output=True,
                    text=True,
                    check=True
                )

                # Push the commit
                push_result = subprocess.run(
                    ["git", "push", "origin", current_branch],
                    cwd=str(local_path),
                    capture_output=True,
                    text=True,
                    check=False
                )

                logging.debug("=" * 80)
                logging.debug("Push output:")
                if push_result.stdout:
                    logging.debug(push_result.stdout)
                if push_result.stderr:
                    logging.debug(push_result.stderr)
                logging.debug("=" * 80)

                if push_result.returncode == 0:
                    logging.debug("Successfully pushed lint fix commit to remote")
                else:
                    logging.debug(f"Push completed with return code: {push_result.returncode}")
            else:
                logging.debug("No GitHub token available, skipping push")
        else:
            logging.debug("Could not parse git_url, skipping push")
    else:
        logging.debug("No formatting changes to commit")

    # Upload the modified directory back to Dir
    formatted_dir = await Dir.from_local(local_path)
    return formatted_dir

@env.task
async def commit_code_review(file_dir: Dir) -> str:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    """
    Perform code review on PR changes using LangChain and OpenAI

    Args:
        file_dir: Directory containing the repository code
        openai_api_key: OpenAI API key for LLM access

    Returns:
        Code review suggestions as a string
    """
    # Download Dir to local path to get git diff
    local_path = await file_dir.download()
    logging.debug(f"Performing code review on: {local_path}")

    try:
        # Get the base branch (usually main or master)
        # First, check which branch we're on
        current_branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(local_path),
            capture_output=True,
            text=True,
            check=True
        )
        current_branch = current_branch_result.stdout.strip()
        logging.debug(f"Current branch: {current_branch}")

        # Get the diff against the base branch using merge-base
        # This ensures we only get changes from this PR, not other commits
        diff_result = None
        base_branch_name = None
        for base_branch in ["origin/main", "origin/master"]:
            try:
                # Find the merge base (common ancestor)
                merge_base_result = subprocess.run(
                    ["git", "merge-base", current_branch, base_branch],
                    cwd=str(local_path),
                    capture_output=True,
                    text=True,
                    check=True
                )
                merge_base = merge_base_result.stdout.strip()
                logging.debug(f"Merge base with {base_branch}: {merge_base}")

                # Get diff from merge base to current branch (only PR changes)
                diff_result = subprocess.run(
                    ["git", "diff", merge_base, current_branch],
                    cwd=str(local_path),
                    capture_output=True,
                    text=True,
                    check=True
                )
                if diff_result.stdout.strip():
                    logging.debug(f"Got diff from merge base against {base_branch}")
                    base_branch_name = base_branch
                    break
            except subprocess.CalledProcessError:
                continue

        if not diff_result or not diff_result.stdout.strip():
            logging.debug("No changes found in PR")
            return "No changes to review"

        git_diff = diff_result.stdout
        logging.debug(f"Git diff size: {len(git_diff)} characters")

        # Initialize OpenAI LLM via LangChain
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            logging.debug("No OpenAI API key provided, skipping code review")
            return "Skipped: No OpenAI API key available"

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=openai_api_key
        )

        # Split diff by files to process large diffs
        # Parse diff to get per-file changes
        file_diffs = []
        current_file_diff = []
        current_file_name = None

        for line in git_diff.split('\n'):
            if line.startswith('diff --git'):
                # Save previous file diff
                if current_file_diff:
                    file_diffs.append({
                        'name': current_file_name,
                        'diff': '\n'.join(current_file_diff),
                        'size': len('\n'.join(current_file_diff))
                    })
                # Start new file diff
                current_file_name = line.split(' ')[-1] if ' ' in line else 'unknown'
                current_file_diff = [line]
            else:
                current_file_diff.append(line)

        # Don't forget the last file
        if current_file_diff:
            file_diffs.append({
                'name': current_file_name,
                'diff': '\n'.join(current_file_diff),
                'size': len('\n'.join(current_file_diff))
            })

        logging.debug(f"Split diff into {len(file_diffs)} files")

        # Process files in batches to stay under token limit
        all_reviews = []
        current_batch = []
        current_batch_size = 0
        max_batch_size = 25000  # Leave room for prompt and response

        system_message = SystemMessage(content="""You are an expert code reviewer.
Review the provided git diff and provide constructive feedback focusing on:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance concerns
4. Security vulnerabilities
5. Code readability and maintainability
6. Test coverage

Provide specific, actionable suggestions for improvement. Be concise.""")

        for i, file_info in enumerate(file_diffs):
            # If adding this file would exceed batch size, process current batch
            if current_batch and (current_batch_size + file_info['size'] > max_batch_size):
                # Process current batch
                batch_diff = '\n\n'.join([f['diff'] for f in current_batch])
                file_list = ', '.join([f['name'] for f in current_batch])

                logging.debug(f"Processing batch with {len(current_batch)} files: {file_list}")

                human_message = HumanMessage(content=f"""Please review the following code changes:

```diff
{batch_diff}
```

Provide a code review with specific suggestions for improvement.""")

                response = llm.invoke([system_message, human_message])
                all_reviews.append(f"### Review for: {file_list}\n\n{response.content}")

                # Reset batch
                current_batch = []
                current_batch_size = 0

            # Add file to current batch
            current_batch.append(file_info)
            current_batch_size += file_info['size']

        # Process remaining batch
        if current_batch:
            batch_diff = '\n\n'.join([f['diff'] for f in current_batch])
            file_list = ', '.join([f['name'] for f in current_batch])

            logging.debug(f"Processing final batch with {len(current_batch)} files: {file_list}")

            human_message = HumanMessage(content=f"""Please review the following code changes:

```diff
{batch_diff}
```

Provide a code review with specific suggestions for improvement.""")

            response = llm.invoke([system_message, human_message])
            all_reviews.append(f"### Review for: {file_list}\n\n{response.content}")

        # Combine all reviews
        review_content = "\n\n---\n\n".join(all_reviews)

        # Log the review
        logging.debug("=" * 80)
        logging.debug("Code Review Results:")
        logging.debug("=" * 80)
        logging.debug(review_content)
        logging.debug("=" * 80)

        return review_content

    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to get git diff: {e.stderr}"
        logging.debug(error_msg)
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Failed to perform code review: {str(e)}"
        logging.debug(error_msg)
        return f"Error: {error_msg}"

@env.task
async def generate_pr_description_and_commit(file_dir: Dir, git_url: str, code_review: str = "") -> str:
    """
    Generate PR description using AI and update the PR on GitHub

    Args:
        file_dir: Directory containing the repository code
        git_url: GitHub PR URL (e.g., https://github.com/owner/repo/pull/123)
        code_review: Code review results to include in PR description

    Returns:
        Generated PR description
    """
    # Get GitHub token from environment variable
    github_token = os.getenv("GITHUB_TOKEN", "")
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    import json

    # Download Dir to local path to get git diff
    local_path = await file_dir.download()
    logging.debug(f"Generating PR description for: {local_path}")

    # Extract PR information from git_url
    pr_pattern = r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)'
    pr_match = re.match(pr_pattern, git_url)

    if not pr_match:
        logging.debug(f"Invalid GitHub PR URL: {git_url}")
        return "Error: Invalid GitHub PR URL"

    owner = pr_match.group(1)
    repo = pr_match.group(2)
    pr_number = pr_match.group(3)

    try:
        # Get current branch name
        current_branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(local_path),
            capture_output=True,
            text=True,
            check=True
        )
        current_branch = current_branch_result.stdout.strip()
        logging.debug(f"Current branch: {current_branch}")

        # Get the diff against the base branch using merge-base
        # This ensures we only get changes from this PR, not other commits
        diff_result = None
        for base_branch in ["origin/main", "origin/master"]:
            try:
                # Find the merge base (common ancestor)
                merge_base_result = subprocess.run(
                    ["git", "merge-base", current_branch, base_branch],
                    cwd=str(local_path),
                    capture_output=True,
                    text=True,
                    check=True
                )
                merge_base = merge_base_result.stdout.strip()
                logging.debug(f"Merge base with {base_branch}: {merge_base}")

                # Get diff from merge base to current branch (only PR changes)
                diff_result = subprocess.run(
                    ["git", "diff", merge_base, current_branch],
                    cwd=str(local_path),
                    capture_output=True,
                    text=True,
                    check=True
                )
                if diff_result.stdout.strip():
                    logging.debug(f"Got diff from merge base against {base_branch}")
                    break
            except subprocess.CalledProcessError:
                continue

        if not diff_result or not diff_result.stdout.strip():
            logging.debug("No changes found in PR")
            return "No changes to describe"

        git_diff = diff_result.stdout
        logging.debug(f"Git diff size: {len(git_diff)} characters")

        # Check for OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logging.debug("No OpenAI API key provided, skipping PR description generation")
            return "Skipped: No OpenAI API key available"

        # Initialize OpenAI LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=openai_api_key
        )

        # Handle large diffs intelligently
        if len(git_diff) > 30000:
            logging.debug("Diff is large, using git stat for summary")

            # Get file statistics instead of full diff
            stat_result = subprocess.run(
                ["git", "diff", "--stat", merge_base, current_branch],
                cwd=str(local_path),
                capture_output=True,
                text=True,
                check=True
            )

            # Get list of changed files with short summary
            files_changed = subprocess.run(
                ["git", "diff", "--name-status", merge_base, current_branch],
                cwd=str(local_path),
                capture_output=True,
                text=True,
                check=True
            )

            # Get commit messages in this PR for context
            log_result = subprocess.run(
                ["git", "log", "--oneline", f"{merge_base}..{current_branch}"],
                cwd=str(local_path),
                capture_output=True,
                text=True,
                check=True
            )

            summary_info = f"""## Statistics:
{stat_result.stdout}

## Files Changed:
{files_changed.stdout}

## Commits:
{log_result.stdout}

## Sample of changes (first 10000 chars of diff):
```diff
{git_diff[:10000]}
```
"""

            system_message = SystemMessage(content="""You are an expert at writing clear and concise PR descriptions.
Based on the git statistics, file changes, commits, and sample diff provided, generate a professional PR description that includes:
1. A brief summary (1-2 sentences)
2. What changed (bullet points based on files and commits)
3. Why these changes were made (infer from commits and changes)
4. Any relevant technical details

Format the description in Markdown.""")

            human_message = HumanMessage(content=f"""Please generate a PR description based on the following information:

{summary_info}

Generate a well-structured PR description in Markdown format.""")
        else:
            # Use full diff for smaller changes
            system_message = SystemMessage(content="""You are an expert at writing clear and concise PR descriptions.
Based on the git diff provided, generate a professional PR description that includes:
1. A brief summary (1-2 sentences)
2. What changed (bullet points)
3. Why these changes were made
4. Any relevant technical details

Format the description in Markdown.""")

            human_message = HumanMessage(content=f"""Please generate a PR description for the following code changes:

```diff
{git_diff}
```

Generate a well-structured PR description in Markdown format.""")

        # Generate PR description
        logging.debug("Generating PR description using OpenAI...")
        response = llm.invoke([system_message, human_message])
        pr_description = response.content

        # Add code review section if available
        if code_review and not code_review.startswith("Skipped:") and not code_review.startswith("Error:"):
            pr_description += f"""

---

## 🤖 AI Code Review

<details>
<summary>Click to expand code review results</summary>

{code_review}

</details>

---

*This code review was automatically generated by AI*
"""
            logging.debug("Added code review section to PR description")

        logging.debug("=" * 80)
        logging.debug("Generated PR Description:")
        logging.debug("=" * 80)
        logging.debug(pr_description)
        logging.debug("=" * 80)

        # Update PR description on GitHub using GitHub API
        if not github_token:
            logging.debug("No GitHub token provided, cannot update PR")
            return pr_description

        # Use GitHub API to update PR
        import urllib.request
        import urllib.error

        api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"

        # Prepare the request
        data = json.dumps({"body": pr_description}).encode('utf-8')

        req = urllib.request.Request(
            api_url,
            data=data,
            headers={
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json"
            },
            method="PATCH"
        )

        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                logging.debug(f"Successfully updated PR #{pr_number} description")
                logging.debug(f"PR URL: {result.get('html_url', '')}")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            logging.debug(f"Failed to update PR description: {e.code} - {error_body}")
            return f"Generated description (failed to update): {pr_description}"

        return pr_description

    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to get git diff: {e.stderr}"
        logging.debug(error_msg)
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Failed to generate PR description: {str(e)}"
        logging.debug(error_msg)
        return f"Error: {error_msg}"



@env.task
async def flyte_sdk_workflow(git_url: str, files_dir: Dir = None) -> Dir:
    """
    Run flyte-sdk specific workflow tasks in parallel

    Args:
        git_url: Git URL of the repository
        files_dir: Directory containing the repository code

    Returns:
        Dir object containing the formatted repository
    """
    logging.debug("Running lint fix and unit tests in parallel...")

    # Run both tasks in parallel
    # formatted_dir, _ = await asyncio.gather(
    #     flyte_sdk_lint_fix(files_dir),
    #     flyte_sdk_unit_test(files_dir)
    # )
    await flyte_sdk_unit_test(files_dir)
    files_dir = await flyte_sdk_lint_fix(files_dir)
    files_dir = await commit_changes(files_dir)
    return files_dir


@env.task
async def wf(git_url: str):
    files_dir = await pull_pr_branch(git_url)
    if git_url.startswith("https://github.com/flyteorg/flyte-sdk"):
        files_dir = await flyte_sdk_workflow(git_url, files_dir)

    # Run code review and get results
    code_review_result = await commit_code_review(files_dir)

    # Generate PR description with code review included
    await generate_pr_description_and_commit(files_dir, git_url, code_review_result)


if __name__ == "__main__":
    # Configure logging for local execution (already configured at module level for Flyte)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

    flyte.init_from_config()
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if openai_api_key:
        logging.debug("OpenAI API key found in environment")
    else:
        logging.debug("No OpenAI API key found in environment")

    run = flyte.with_runcontext(
        log_level=logging.DEBUG,
        overwrite_cache=True,
        interruptible=False,
    ).run(wf, git_url="https://github.com/flyteorg/flyte-sdk/pull/341")
    logging.debug(f"Run name: {run.name}")
    logging.debug(f"Run URL: {run.url}")
    run.wait()
