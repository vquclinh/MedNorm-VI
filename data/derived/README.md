# data/derived/ — untracked derived artifacts

Deterministically transformed resources (normalized tables, snapshot indexes,
label-mapped corpora) live here locally. Git-ignored. Never committed.

Every derived artifact should be described by a `DerivedArtifactRecord` inside the
source resource's manifest (transformation script + checksum), so it can be
rebuilt deterministically from the raw resource.
