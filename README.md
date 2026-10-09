# ApertusHack - Fact Checking

Fact-checking claims about Swiss popular votes against the official voting booklets, using the Apertus model. This is our entry for [Hack Apertus](https://hackapertus.ch/), Track 2A, OST challenge _Multilingual Natural Language Inference over Swiss Official Voting Booklets_.

The repository follows the [HackApertus project template](https://github.com/HackApertus/project-template): the project is in [`track_2a/`](track_2a/), and that folder is its root. Start with [`track_2a/README.md`](track_2a/README.md).

```bash
cd track_2a
make run      # builds the Docker image and decides the sample cases; needs API_KEY, see .env.example
```

- **Docker image:** `ghcr.io/pgied/apertushack-fact-checking-pub:latest`, built and published by [`.github/workflows/image.yml`](.github/workflows/image.yml).
- **The challenge:** [Track 2A: OST](https://hackapertus.notion.site/3deb4fec112a80258fd2ddc61616011d), with its [Solution API Guide](https://hackapertus.notion.site/ef5b4fec112a834da43101ede56300f5) for the input, the output and the Docker image.
- **License:** the code under [Apache-2.0](LICENSE), the documentation under CC-BY-4.0, as section 6 of the Hack Apertus [Terms & Conditions](https://hackapertus.ch/terms-and-conditions) asks.
