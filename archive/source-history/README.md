# Source history boundary

This directory contains completed experimental implementations whose original
names are part of immutable research history.

Files here are not active runtime, training, packaging, or test entry points.
They are preserved byte-for-byte when a locked evidence record refers to their
old names. New implementation code belongs under `src/Pet-ReID-IMAG` and must
use deployment roles or capability names instead of project-generation labels.

The `high-resolution-candidate-lifecycle` directory preserves the completed
protocol-build, candidate-lock, one-shot blind evaluation, export, and package
tools for the locked spatial-detail release. The `external-joint-lifecycle`
directory preserves the ended legacy-generation acceptance comparison. Neither
directory is searched for active commands or collected as part of the test suite.

The `unified-semantic-candidate-lifecycle` directory preserves its one-time
release-lock utility. `upstream-reproduction-configs` retains redundant
author-era configuration names that are useful only when comparing historical
commands; active equivalents use architecture and input-size names.

Unlike the rest of `archive/`, this source-history subtree is intentionally
tracked by Git so moving a completed implementation here does not silently turn
into an unrecoverable source deletion.

relocations.json maps source paths recorded by immutable candidate evidence to
their byte-identical archived locations. The workspace metadata audit requires
every recorded source path that no longer exists in active source to have
exactly one mapped archive file with the locked SHA-256.

Published package paths, protocol records, acceptance JSON, model fingerprints,
API route versions, and serialization schema versions remain unchanged.
