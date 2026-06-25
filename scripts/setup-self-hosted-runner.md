# Self-hosted GitHub Actions runner for this repo

Use this when your Databricks workspace has an IP ACL that blocks
GitHub-hosted runners (the default `runs-on: ubuntu-latest`). The workflow
in `.github/workflows/databricks-cicd.yml` is configured for
`runs-on: [self-hosted, fsi-fraud-cicd]` — register a runner with the
`fsi-fraud-cicd` label on any host whose egress IP is on the workspace
allowlist (your Mac on Databricks VPN works; a small EC2/GCE box also
works).

## When NOT to do this

For a real Databricks-customer repo, move the repo into the
`databricks-field-eng` GitHub org and use the IT-managed runners instead.
That's the supported path:

```yaml
runs-on:
  group: databricks-field-eng-protected-runner-group
  labels: linux-ubuntu-latest
```

IT-managed runners are pre-approved by FE's workspace ACL — no per-repo
runner registration required. See:
https://databricks.atlassian.net/wiki/spaces/UN/pages/4045112025

Self-hosted runners are the right call only when the repo lives outside
the databricks GH orgs (e.g. a personal-namespace demo like
github.com/tiantan32/fsi-fraud-detection-dab).

## Setup (macOS / Linux)

1. **Provision a host whose IP is on the workspace allowlist.** Easiest:
   your laptop on Databricks VPN. For an always-on demo, an EC2 t3.small
   with the VPN client also works.

2. **Confirm the host can reach the workspace:**
   ```bash
   curl -sS -H "Authorization: Bearer $DATABRICKS_TOKEN" \
     https://fe-vm-ttan-vm.cloud.databricks.com/api/2.0/preview/scim/v2/Me \
     | jq .userName
   # Should print: tian.tan@databricks.com  (or whoever the PAT belongs to)
   ```

3. **Register the runner.** From the repo settings:
   `Settings -> Actions -> Runners -> New self-hosted runner`. GitHub gives
   you the exact commands; the important bit is the `--labels` flag:

   ```bash
   mkdir actions-runner && cd actions-runner
   curl -o actions-runner-osx-arm64.tar.gz -L \
     https://github.com/actions/runner/releases/download/v2.319.1/actions-runner-osx-arm64-2.319.1.tar.gz
   tar xzf actions-runner-osx-arm64.tar.gz
   ./config.sh --url https://github.com/tiantan32/fsi-fraud-detection-dab \
               --token <one-time-token-from-GH-UI> \
               --labels fsi-fraud-cicd \
               --unattended
   ./run.sh
   ```

   Leave `run.sh` running in tmux/screen, or install as a service:
   ```bash
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```

4. **Verify** the runner shows up under
   `Settings -> Actions -> Runners` as Idle with the `fsi-fraud-cicd` label.

5. **Trigger CI** with a small commit; the workflow's `validate-staging`
   job should pick up the self-hosted runner and the `databricks bundle
   validate --target staging` step should succeed (no 403 IP ACL error).

## Security notes

- Self-hosted runners execute arbitrary CI code from the repo. Only register
  one for repos you trust. Don't share the runner across orgs.
- The runner reuses the host's network — if you put it on a CSP VM,
  restrict the VM's egress to the workspace's domain.
- The Databricks PAT in `DATABRICKS_TOKEN_STAGING` / `_PROD` repo secrets
  is what auths the bundle CLI; the runner host doesn't need any extra
  Databricks credential.
