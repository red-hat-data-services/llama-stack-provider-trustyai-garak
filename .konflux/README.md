# Downstream base image

`Dockerfile.konflux` contains the authoritative AIPCC base pin in the default value of `BASE_IMAGE`.
The image reference includes a tag and a multi-architecture digest.
The stage argument supplies the same reference to `com.redhat.aiplatform.image`.

The build tests PyArrow and provider imports after dependency installation.
An import failure stops the image build on the affected architecture.

`cpu-ubi9.conf` is an empty compatibility file for older PipelineRuns.
It supplies no arguments and cannot override the Dockerfile default.
The central pipeline update removes its `build-args-file` reference.

The upstream `Containerfile` remains separate from this downstream build.
The migration preserves the existing base digest and changes no Python dependency pins.

## Rollout

1. Merge the validation workflow PR.
2. Deploy its checks to the selected release branches.
3. Ask an administrator to require the intended CI and Konflux checks.
4. Make sure that the bot approval policy remains compatible with the organization rules.
5. Merge the Garak PR before the central PipelineRun update.
6. Merge the corresponding `konflux-central` PR.
7. Make sure that the synchronized PipelineRun no longer specifies `build-args-file`.
8. Remove the empty compatibility file after no active pipeline references it.

Do not restore a `BASE_IMAGE` assignment in the compatibility file.
That assignment overrides the pin that Renovate updates in the Dockerfile.

## Renovate policy

`.github/renovate.json` enables automatic merges for Renovate updates, including major versions.
Renovate waits for successful reported checks and performs the merge itself.
The policy does not change approval requirements or the team-managed Jira gate.

The central copy is `renovate/garak-renovate.json` in `konflux-central`.
Its sync mapping uses `.github/renovate.json` as the target file.
Both copies use the same policy and preserve MintMaker's default managers and branch configuration.

Required-check rules enforce test coverage when a job is absent or skipped.
A successful reported check does not prove that an absent test ran.

Frozen branches need separate Dockerfile and workflow backports.
Each backport retains the approved image family and digest for its release stream.
