# Weekly Numerai Submission Prompt

Use the custom `numerai-weekly` MCP server and Numerai's official MCP server to run the weekly operational loop.

## Workflow

1. Weekly Retrain: Trigger `run_weekly_retrain`.
2. Monitoring: Poll `check_retrain_status` until the job completes.
3. Evaluation: Review the latest config with `get_training_summary`.
4. Data Drift Analysis: Compare the selected features versus the prior run with `compare_weekly_features`.
5. Outputs: Generate a markdown and HTML report with `generate_weekly_report`.
6. Submissions: Use the official Numerai MCP to upload the packaged model artifact.