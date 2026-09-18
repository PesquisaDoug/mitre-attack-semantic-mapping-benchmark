# Data

This project downloads the official Enterprise ATT&CK STIX bundle from:

https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json

The local raw bundle is stored under `data/raw/enterprise-attack.json` and is
ignored by Git. Reproducibility metadata, including SHA-256, file size, STIX
object count and latest object modification timestamp, is written to
`data/data_manifest.json`.

The generated benchmark files are:

- `data/techniques.csv`: active Enterprise ATT&CK techniques and sub-techniques.
- `data/queries.csv`: procedure-description queries built from active STIX
  `uses` relationships with non-empty descriptions.

ATT&CK data is subject to MITRE's official terms of use:

https://attack.mitre.org/resources/terms-of-use/
