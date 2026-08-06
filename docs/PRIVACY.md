# Privacy Notice

## What the application does by default

The application reads camera frames and performs detection and analysis
locally on the machine running it. Frames are held in memory long enough to
process and display them. The repository's default runtime does not save
frames, write detection or demographic results to a log, or intentionally
upload webcam content to a remote service.

Closing the application releases its camera handle and in-memory frames.
Operating-system camera drivers, screen recording tools, modified builds, and
third-party software are outside this repository's control.

## Network access and model downloads

Local frame processing does not mean that every installation step is offline.
Package installation contacts package repositories. This suite rejects a
missing YOLO checkpoint before Ultralytics can fetch it. DeepFace can download
backend weights only after the operator passes `--allow-model-downloads`.
Those upstream services receive ordinary network
metadata such as an IP address and request time. The project does not intend
to include webcam frames in those requests, but third-party library behavior
and privacy terms must be reviewed for the versions deployed.

For a strict offline or high-privacy environment, download dependencies and
models through a controlled build process, verify their hashes and licenses,
then block application network access during inference.

## Age, gender, and emotion estimates

Age, gender presentation, and emotion outputs are probabilistic model
estimates. They can be inaccurate, biased across populations, unstable across
lighting or pose, and misleading. A model label is not a person's gender
identity, actual age, mental state, intent, health, or character.

Attribute analysis should be disabled unless the operator deliberately opts
in for a legitimate, documented use. Obtain informed consent from every
person likely to enter the camera view, provide a visible camera indicator,
and offer a practical way to decline. Take particular care with children,
bystanders, workplaces, schools, and public spaces.

Do not use these estimates to make or materially influence decisions about
employment, education, housing, credit, insurance, health care, policing,
immigration, legal rights, access to services, or any other high-impact
matter. Do not use them for covert surveillance or identity claims. Operators
are responsible for applicable privacy, biometric, employment, consumer, and
child-protection laws.

## Collection and retention

The privacy-preserving default is no retained camera or attribute data. If an
operator modifies the application to save frames, recordings, identifiers, or
inferences, that operator determines the purpose and means of processing and
is responsible for an appropriate privacy notice and lawful basis.

Before enabling retention:

- collect the minimum data needed for a stated purpose;
- avoid raw frames when anonymous aggregate data is sufficient;
- define and enforce a short deletion deadline;
- encrypt stored data and restrict access;
- never place personal data in source control or ordinary diagnostic logs;
- document sharing, cross-border transfers, and subprocessors; and
- provide access and deletion processes where required.

Backups and exported reports must follow the same retention and deletion
rules. Remove test captures and legacy logs before sharing a repository or
support bundle.

## Security and contact

Local processing reduces network exposure but does not eliminate privacy
risk. Keep the device patched, limit camera and file permissions, verify model
files, and follow the [security policy](SECURITY.md). Privacy concerns can be sent to
**saad@triangletech.com.bd**; do not attach captured images or other personal
data to the initial message.
