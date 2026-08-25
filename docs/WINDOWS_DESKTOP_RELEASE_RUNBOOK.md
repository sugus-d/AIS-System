# Windows Desktop Release Status

## Completed release-path work

- Electron main process now creates a per-launch random service token, assigns dynamic loopback ports, waits for all three local services, and terminates complete Windows process trees during shutdown.
- Production resources now resolve from the electron-builder layout: `app.asar` holds the Electron app, while algorithm, annotation, migrations, and deployment resources are read from `resources`.
- The staged Node service now serves the renderer from `AIS_RENDERER_ROOT`, binds to `127.0.0.1`, and closes its HTTP listener before exiting.
- The algorithm API exposes `/health` and rejects requests without `x-ais-service-token`; the Node algorithm client forwards that token.
- The release assembler now packages the renderer separately, requires a pinned Python lock before staging, and writes `manifest.json` plus a SHA-256 `release-report.json`.
- The runtime verifier now exercises dynamic loopback ports, startup health checks, SQLite initialization, algorithm-token enforcement, renderer presence, manifest/report parsing, and Windows process-tree cleanup.
- NSIS is configured as a per-user x64 installer; uninstall preserves user data.

## Required release-runner prerequisites

The build intentionally stops until these prerequisites are supplied:

1. Generate and commit the reviewed production dependency lock:

   ```powershell
   pnpm runtime:lock
   ```

2. Set `AIS_RUNTIME_PYTHON_HOME` to a clean, pinned CPython 3.11 runtime directory. It must contain `python.exe` and must not be a project virtual environment.

3. Ensure `pnpm install --frozen-lockfile` succeeds on the release runner. The current repository checkout has unresolved UI dependency/typecheck problems that must be fixed before signing a release.

4. Configure Authenticode signing only through CI secret management or an enterprise certificate service. Do not place certificate files or private keys in the repository, `.release`, or installer output.

## Build and verification sequence

```powershell
$env:AIS_RUNTIME_PYTHON_HOME = 'C:\release-tools\python311'
pnpm runtime:lock
pnpm build:win
```

`pnpm build:win` stages `.release`, validates the staged runtime, then invokes electron-builder to create `release\AIS-<version>-Setup.exe`.

## Remaining mandatory acceptance work

- Run the installer in a clean Windows x64 VM with no Node.js, Python, source tree, or developer tools.
- Execute a representative PLY import, real algorithm analysis, report generation, annotation session, restart, upgrade, uninstall, and user-data retention test.
- Verify Authenticode signatures on the installer and installed executable.
- Record the clean-machine results and artifact hashes alongside the signed release.
