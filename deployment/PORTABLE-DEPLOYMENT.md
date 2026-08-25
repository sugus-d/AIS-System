# Portable deployment

Before distributing a package, replace `initial-admin.json` with the deployment-specific administrator account and a password of at least 12 characters.

On the first successful start, the application imports this account into SQLite and renames the file to `initial-admin.json.consumed`. The database, files, results, and logs are stored below the Windows per-user application-data directory and are not replaced by upgrades.

The local backup command creates a version 2 archive containing the SQLite database, scans, algorithm outputs, and annotation data. Every archived file is recorded with its size and SHA-256 digest. Restore validates every digest before scheduling the restore for the next application start; archives made by older versions without per-file integrity data are intentionally rejected.

Interrupted `pending` and `running` analysis tasks are automatically resumed when the application starts. Desktop service logs are stored under `ais-data/logs`, rotate at 10 MB by default, and retain three archived log files. Deployments may override the rotation limit with `AIS_LOG_MAX_BYTES`.
