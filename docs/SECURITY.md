# Security Policy

## Supported versions

Security fixes are made only to the latest published revision.

| Version | Security support |
| --- | --- |
| Latest commit on the default branch | Supported |
| Older commits, archived copies, and forks | Not supported; reproduce on or upgrade to the latest revision |

## Report a vulnerability privately

Do not open a public issue for an unpatched vulnerability. Send the initial
report to **saad@triangletech.com.bd** with the subject
**[SECURITY] AI Detection Suite vulnerability**.

This is the project's temporary reporting address. Maintainers should replace
it with a dedicated, monitored security mailbox before a production deployment.

Include:

- the affected revision, file, feature, and operating system;
- the impact and conditions required to trigger it;
- minimal, non-destructive reproduction steps; and
- a suggested remediation, if known.

Do not include credentials, API keys, personal data, captured camera images,
malicious checkpoint files, or a weaponized exploit in the first message.
Ask for an encrypted transfer channel if sensitive artifacts are necessary.
Please allow the maintainers to investigate and coordinate a fix before
publishing technical details. We aim to acknowledge complete reports within
five business days, but this is not a service-level guarantee.

## Model and checkpoint threat policy

PyTorch checkpoint files can contain pickle-backed objects. Loading a
malicious .pt or .pth file can execute attacker-controlled code with the
application user's permissions. A matching hash establishes file identity; it
does not prove that pickle content is intrinsically safe.

Production code and releases must:

1. Accept only the bundled checkpoint paths recorded in
   [ai_detection_suite/models/model-manifest.json](../ai_detection_suite/models/model-manifest.json).
2. Validate the canonical path, file type, byte size, and SHA-256 digest with
   ai_detection_suite.model_security before passing a path to a model loader.
3. Reject symlinks, files outside the configured model trust root, unexpected
   extensions, changed files, and unlisted models.
4. Obtain checkpoints through a reviewed release process, store them in a
   write-restricted location, and update manifest hashes only after provenance
   review.
5. Run inference with least privilege and without unnecessary filesystem or
   network access.

Runtime downloads and user-supplied checkpoint URLs are not trusted
production inputs. The explicit unverified-model override is for isolated
development only. If a custom model is required, acquire and inspect it
offline, verify its origin and license, scan it, and prefer a format that
cannot embed Python objects (such as a validated ONNX or safetensors artifact)
before introducing it to the application.

Reports involving model poisoning, unsafe deserialization, dependency
compromise, path traversal, privacy disclosure, camera access, or
resource-exhaustion denial of service are in scope.

## Dependencies and secrets

- Install from a reviewed, version-pinned lock file and verify package hashes
  in production builds. Reassess dependencies and downloaded model assets
  regularly.
- Do not commit passwords, tokens, private keys, credentials, or sensitive
  environment files. Inject secrets at runtime from an operating-system or
  deployment secret store.
- Treat a committed secret as compromised: revoke or rotate it first, then
  remove it from current files and repository history.
- Do not log frames, inferred attributes, secrets, absolute user paths, or
  other personal data. Keep errors and diagnostics bounded and redacted.

When discussing a fixed issue publicly, omit exploit details until affected
users have had a reasonable opportunity to update.
