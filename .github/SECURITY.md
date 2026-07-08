# Security Policy

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report privately through GitHub's
[private vulnerability reporting](https://github.com/AI-ModCon/dsagt/security/advisories/new)
(the **Security** tab → *Report a vulnerability*). If that isn't available to
you, email the maintainers at **aaron.tuor@pnnl.gov**.

Please include:

- a description of the issue and its impact,
- steps to reproduce (or a proof of concept),
- affected version / commit, and
- any suggested remediation.

The maintainers will acknowledge your report, keep you updated on progress, and
credit you in the fix unless you prefer to remain anonymous.

## Scope

DSAgt executes CLI codes and installs skills that the agent registers, and it
runs code from external skill catalogs the user chooses to sync. It does **not**
sandbox that code — see the "Risks" section of [agent-card.md](../agent-card.md).
Reports most relevant to this project include: path-traversal or arbitrary
file write/delete from untrusted skill or code specs, injection via indexed
knowledge-base documents, and unintended handling of credentials.

## Supported versions

DSAgt is pre-1.0 and dev-stage; security fixes are applied to the latest
development line (currently the `0.2.x` series on `main`).
