# Security Policy

## Reporting

Please report suspected vulnerabilities privately to the maintainers instead of
opening a public issue.

Include:

- A concise description of the issue
- Steps to reproduce or a proof of concept
- Affected version or commit, when known
- Any known impact on API keys, benchmark data, or generated reports

## Secret Handling

Never commit `.env`, provider API keys, account identifiers, private result
files, or unpublished benchmark data. If a credential is committed or shared,
rotate it with the provider immediately.

## Scope

Security-sensitive areas include:

- API key loading and error handling
- Generated HTML reports
- Static-site export files
- CSV parsing and report regeneration from local session files
