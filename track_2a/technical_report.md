# Technical report — `project name`

- **Track:** `Track 2A — OST: Multilingual Natural Language Inference over Swiss Official Voting Booklets`
- **Event:** Online
- **Team:** `team name` — `member`, `member`, `Patrick Giedemann`
- **Demo:** the Docker image `ghcr.io/pgied/apertushack-fact-checking-pub:latest` (no demo video required)
- **Code:** the beginner task is [`gied_beginner_scoped`](src/strategies/gied_beginner_scoped/strategy.py), the advanced task is [`gied_strict_reasoned`](src/strategies/gied_strict_reasoned/strategy.py); the shared retrieval code is in [`src/strategies/_gied/`](src/strategies/_gied/).

## 1. Summary

Every Swiss popular vote comes with an official voting booklet in German, French and Italian. The task is natural language inference (NLI; Dagan et al., 2006; Williams et al., 2018): does the booklet support a claim about the vote (entailment), contradict it (contradiction) or neither (neutral)? Two thirds of the claims are in another language than the booklet, as in cross-lingual NLI (Conneau et al., 2018). The beginner task gives a section of the booklet, the reference. The advanced task gives the whole booklet PDF and asks for the evidence too.

Both of our solutions **retrieve, then verify**, as evidence-based fact checking does (Thorne et al., 2018; Guo et al., 2022): local code finds the few sentences that bear on the claim, and Apertus v1.5 70B judges only those. On all 1,488 cases of the public dataset:

- **Beginner task:** macro-F1 **0.987**, with 834 tokens and 0.6 ms of processing time per case. The baseline prompt reached 0.788, with about 1,900 tokens.
- **Advanced task:** macro-F1 **0.974** and evidence Hit@5 **0.979**, with 1,157 tokens and 3.6 ms per case. Full-document prompting reached 0.70–0.73 and 0.13 on 150 of these cases, with about 34,770 tokens per case.

## 2. Architecture

The CLI, [`src/entrypoint.py`](src/entrypoint.py), runs in our Docker image (linux/amd64, CPU only). It reads the cases, sets up the strategy of each task, decides 8 cases at a time in threads, and writes one JSON line per case. Everything but Apertus runs in the container: PDFium (`pypdfium2`) reads the PDFs, and BM25 is our own Python code. There is no OCR, no embedding model and no other model. A case sends one request to Apertus if the claim is in the document's language, otherwise two: a translation, then the decision.

| Step                   | Beginner task: reference text                                      | Advanced task: booklet PDF                                                                    |
| ---------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| 1. Read                | The reference as given                                             | The booklet's tagged structure: headings, paragraphs, reading order                           |
| 2. Index               | Sentences, ranked by BM25                                          | Sentences with their page and headings, ranked by BM25                                        |
| 3. Narrow              | —                                                                  | Only the pages of the claim's vote                                                            |
| 4. Cross-lingual claim | Apertus translates it; the prompt shows both versions              | The same                                                                                      |
| 5. Select              | The 6 best sentences with their neighbours, about 2,000 characters | The 8 best sentences with their neighbours, about 2,000 characters                            |
| 6. Decide              | Apertus explains in at most three sentences, then gives the label  | The same, and Apertus names the excerpts that decide it                                       |
| 7. Evidence            | The reference or the part shown, page `null`; none for neutral (not scored) | Five verbatim sentences of the booklet with their pages                              |

### 2.1 Shared by both tasks: the sentence index and the translation

**The sentence index.** The text is split into sentences, never across a paragraph or a page. Abbreviations such as "z. B." do not end a sentence, and one under 40 characters joins the next. Sentences are the unit because the evidence has to be short and verbatim, and small units fit more of the relevant text into a fixed context (Chen et al., 2024). BM25 ranks them for the claim (k1 = 1.2, b = 0.75; Robertson and Zaragoza, 2009):

- **Words** are lower-cased, stripped of accents and cut to their first 7 letters. Cutting words to a fixed prefix is a language-independent stand-in for stemming (McNamee et al., 2009), which suits German compounds as well as French and Italian inflection. 7 letters found more evidence than 5, 6 or whole words.
- **Neighbours:** a sentence also gets 30 % of its better neighbour's score.

**The translation.** Lexical search needs the claim in the document's language. So Apertus first translates a claim in another language, and the search uses the translation. This is translate-test (Conneau et al., 2018; Artetxe et al., 2023), with the model translating for itself (self-translate; Etxaniz et al., 2024).

### 2.2 Beginner task: a scoped reference

**1. Translate** a claim in another language into the reference's language (Section 2.1). The prompt shows the claim, then `THE CLAIM IN FRENCH (a translation, to compare it with the reference):` and the translation.

**2. Scope the reference.** References are whole booklet sections: 3,700 characters in the median, 7,100 on average, up to 25,000, and most of the input. For the 1,225 references longer than 2,000 characters, we keep the 6 sentences that match the claim best (Section 2.1, with the translated claim) and the sentence before and after each. They stay in the reference's order, about 2,000 characters in all, with "[…]" for each gap.

**3. Decide.** The label definitions put neutral first and allow contradiction only with a sentence to quote, after the idea of supporting answers with verified quotes (Menick et al., 2022):

> - **neutral:** the reference neither supports nor contradicts the claim
> - **contradiction:** the reference says the opposite of the claim, and you can quote the sentence that does
> - **entailment:** the reference supports the claim
>
> If the reference does not mention what the claim says, or is about another topic, the label is "neutral", not "contradiction": a claim the reference does not support is not contradicted by it.

The model first explains in at most three sentences what the reference says about the claim. Then it writes `Label: <word>` on a last line, which is the line we read. Reasoning before the answer helps (Wei et al., 2022), and an explanation written after the label cannot change it.

**Why these steps.** Almost all of the baseline's errors were neutral claims answered contradiction: 274 of 498. The dataset's neutral claims mostly add a detail the reference does not give. We added one step at a time:

1. **Neutral first, with reasoning:** 140 such errors were left. In 86 of them, the model's own explanation said the reference "does not mention" the claim before it answered contradiction.
2. **The `quote` definitions,** which call that neutral: 46 were left.
3. **Scoping** fixed 44 of the 52 errors left after the translation step, 39 of them neutral claims answered contradiction. It made 12 new ones, 5 of them contradictions whose deciding sentence was cut away.

This matches the finding that irrelevant context distracts language models (Shi et al., 2023).

### 2.3 Advanced task: retrieve, then verify

**1. Read the booklet** once, when a case first needs it. 59 of the 60 booklets are tagged PDFs (ISO 32000-2 §14.8; PDF/UA, ISO 14289-1). Their structure tree gives the headings, the paragraphs and the reading order, without running headers and page numbers. PDFium reads the text of each tagged element. The voting text is placed untagged, so a page with much untagged text takes its body text from the text layer. Of three readers, the tags kept the most of each gold passage: 97.9 % on average at 3 ms per page. The plain text layer kept 96.0 % at 1 ms per page, and Docling (Auer et al., 2024) 95.9 % at 157 ms per page on a CPU.

**2. Index the sentences** of the body text (Section 2.1). Each sentence also keeps its page and the two headings above it.

**3. Narrow to the claim's vote.** A booklet covers one to five votes, and each case names its vote. Running headers such as "Erste Vorlage: …" or "Deuxième objet : …" assign the pages of the booklet's detailed part to the votes. We keep the vote whose pages share the most rare words and word pairs with the case's vote title. These pages are a third of a booklet on average, and they hold all gold pages in 989 of the 990 cases with a gold entailment or contradiction. Claims about the voting text or the summary read alike for every vote, so they had often been judged against another vote's text.

**4. Search the supplied booklet.** A claim in the booklet's language is searched as it is, and a claim in another language with its translation (Section 2.1), within the pages of the claim's vote. We read only the booklet the case supplies. Every booklet is also published in the other two languages with the same pages, and searching the edition in the claim's language as well raised macro-F1 to 0.962 in our runs (Table 3). But that edition is in the input folder only when another case uses it, and the challenge asks to judge against the supplied source, so we left it out.

**5. Build the excerpts.** The 8 best sentences become numbered excerpts, best first, each with the sentence before and after it on its page, up to about 2,000 characters in all. Each excerpt names its page and section, for example `[3] (p. 12, Argumente › Zu strikte Vorgaben) …`. That is a few percent of a booklet, which has 23,000 to 270,000 characters. Long and distracting contexts lower accuracy (Liu et al., 2024; Shi et al., 2023), and retrieval with a short context can match long-context models (Xu et al., 2024).

**6. Decide as in the beginner task** (Section 2.2). Apertus gets the vote, the excerpts, the claim and its translation, and the `quote` definitions, with "booklet" in place of "reference". It first explains in at most three sentences what the excerpts say about the claim, quoting the sentence that says the opposite for a contradiction. Then it writes `Label: <word>` and, on a last line, `Excerpts: <numbers>`, the excerpts that decide it. Only these two lines are read, since the reasoning may name other labels and numbers. The prompt also says that only the booklet counts, not the model's own knowledge.

We first asked for JSON only, `{"label": …, "excerpts": [numbers]}`, with a definition of neutral that covered a claim with a detail the booklet does not give, such as a number or a date. That answered 42 contradictions neutral, often with the opposite in the excerpts. The token limit was not the cause: given 250 tokens instead of 40, the model answered neutral in all 42 again, since it wrote the label first and any explanation after it. Deciding as in the beginner task cut them to 4 (Table 3).

**7. Return the evidence:** five sentences of the booklet, verbatim in its language, with their 1-based pages. They come in this order:

1. The sentences found that are also in an excerpt the model cited.
2. The other sentences found, best first.
3. The rest of the cited excerpts.

The vote's title, which starts every gold passage, and fragments under 40 characters are never returned. A neutral answer gets evidence too, since the challenge scores the evidence of every case whose gold label is entailment or contradiction.

## 3. Use of Apertus

- **Model:** `swiss-ai/Apertus-v1.5-70B` (Project Apertus, 2025), at temperature 0. The 8B model served during development and scored lower.
- **How:** inference only, prompted zero-shot, with no fine-tuning. Apertus does two things:
  - **Translation** of a cross-lingual claim, at most 200 output tokens.
  - **The decision:** in both tasks at most three sentences of reasoning, then the label: at most 200 output tokens in the beginner task, and 250 in the advanced task, whose answer also names the excerpts.
- **Where:** a hosted endpoint, called through the OpenAI-compatible chat API at `BASE_URL`: the CSCS Inference API during development, the organisers' proxy at the evaluation. 8 requests run in parallel.

## 4. Data

- **Dataset:** `OSTswiss/MNLIoverSwissVotingBooklets`, version `v1.1`, pinned to one revision (OST, 2026). It has 1,488 rows, each giving a beginner and an advanced case with the same claim and label. The labels are balanced, the claims are split evenly between German, French and Italian, and two thirds of the cases are cross-lingual. The dataset card gives its licence.
- **Booklets:** the 60 official booklets the dataset refers to, downloaded from the Federal Chancellery (bk.admin.ch). They are public and are not redistributed here, except one for the `make run` sample.
- **Gold pages:** for development only, we locate each reference text in its booklet to measure page hits. The official evidence score matches text only.
- **Personal data:** none.

Swiss voting booklets have also been studied for stance detection (Egli et al., 2023).

## 5. Evaluation

**Metrics.** These are the challenge's own:

- **Macro-F1** over the three labels.
- **Evidence Hit@5,** in the advanced task: one of the first five items, at most 5,000 characters, matches the gold passage with rapidfuzz `partial_ratio` ≥ 90 after normalisation. We compute it as the challenge's `evaluate.py` does.
- **Tokens per case:** input and output, as the API reports them.
- **Processing time per case:** the run's wall-clock time minus the time in which at least one request to Apertus is in flight, divided by the number of cases. We measure it in-process ([`src/timing.py`](src/timing.py)), from when Python starts running our code to the last prediction, on a laptop.

All runs used Apertus v1.5 70B through the CSCS Inference API. The endpoint is not fully deterministic at temperature 0: 4 of 515 labels changed when identical requests were repeated, so differences under about 0.005 are noise.

**Table 1: The solutions against the baselines** (all 1,488 cases, unless noted)

| Task     | System                                           | Macro-F1      | Hit@5     | Tokens per case (in + out) | Processing time per case |
| -------- | ------------------------------------------------ | ------------- | --------- | -------------------------- | ------------------------ |
| Beginner | Baseline: the whole reference, a one-word answer | 0.788         | –         | 1,897 + ≤ 10               | –                        |
| Beginner | **Ours**                                         | **0.987**     | –         | 748 + 86                   | 0.6 ms                   |
| Advanced | Full-document prompting (150 cases, 2 runs)      | 0.701 / 0.727 | 0.128     | 34,762 + ≤ 10              | –                        |
| Advanced | Ours, on the same 150 cases                      | 0.978         | 1.000     | 1,048 + 102                | –                        |
| Advanced | **Ours**                                         | **0.974**     | **0.979** | 1,053 + 104                | 3.6 ms                   |

**Table 2: By language** (all 1,488 cases; the language of the claim, of the document, and whether they differ)

| Cases                                 | Number          | Beginner macro-F1     | Advanced macro-F1     | Advanced Hit@5        |
| ------------------------------------- | --------------- | --------------------- | --------------------- | --------------------- |
| All                                   | 1,488           | 0.987                 | 0.974                 | 0.979                 |
| Claim in German / French / Italian    | 495 / 497 / 496 | 0.980 / 0.990 / 0.990 | 0.966 / 0.978 / 0.978 | 0.967 / 0.985 / 0.985 |
| Document in German / French / Italian | 455 / 569 / 464 | 0.985 / 0.989 / 0.985 | 0.963 / 0.972 / 0.987 | 0.970 / 0.987 / 0.977 |
| Same language / cross-lingual         | 515 / 973       | 0.990 / 0.985         | 0.958 / 0.982         | 0.988 / 0.974         |

Across the nine pairs of document and claim language, macro-F1 ranges from 0.973 to 1.000 in the beginner task, and from 0.923 (German booklet, German claim) to 0.987 in the advanced task.

**Table 3: What each step added** (all 1,488 cases unless noted)

| Task     | Step                                                                                                | Macro-F1  | Hit@5 | Input tokens per case |
| -------- | --------------------------------------------------------------------------------------------------- | --------- | ----- | --------------------- |
| Beginner | Baseline prompt                                                                                     | 0.788     | –     | 1,897                 |
| Beginner | Neutral first, reasoning before the label                                                           | 0.901     | –     | 1,952                 |
| Beginner | `quote` definitions, reasoning before the label                                                     | 0.962     | –     | 1,984                 |
| Beginner | + the translation of a cross-lingual claim                                                          | 0.965     | –     | 2,092                 |
| Beginner | + only the sentences around the claim (the solution)                                                | 0.987     | –     | 748                   |
| Advanced | Full-document prompting (150 cases)                                                                 | 0.70–0.73 | 0.13  | 34,762                |
| Advanced | Only the vote's pages; the model quotes sentences, which are looked up in the PDF                   | 0.850     | 0.485 | 11,553                |
| Advanced | Retrieve, then verify: tags, BM25, the edition in the claim's language, a narrow contradiction      | 0.927     | 0.957 | 960                   |
| Advanced | + the search with the translated claim, excerpts of both editions                                   | 0.953     | 0.987 | 1,471                 |
| Advanced | + only the pages of the claim's vote                                                                | 0.962     | 0.988 | 1,481                 |
| Advanced | Only the supplied booklet: the translated claim alone, shown to the model                           | 0.951     | 0.978 | 1,077                 |
| Advanced | + deciding as in the beginner task, reasoning first (the solution)                                  | 0.974     | 0.979 | 1,053                 |

The output adds about 86 tokens per case in the beginner task (the translation and the reasoning), and about 104 in the advanced task (the translation and the reasoning).

**Tried and left out:**

- **Paraphrases:** five paraphrases of the claim, searched alongside it (after Hoyle et al., 2023). In the beginner task they reached 0.979, for 2,513 tokens per case. In the advanced task they reached 0.947 on top of the translation, against 0.951 without, for 25 % more tokens; the rankings were merged by reciprocal rank fusion (Cormack et al., 2009).
- **The booklet's edition in the claim's language** (advanced task): searching it as well, with each sentence found there mapped to the sentence at the same relative position on the same page of the booklet, a positional variant of length-based sentence alignment (Gale and Church, 1993), raised macro-F1 to 0.962 and Hit@5 to 0.988. It is left out for the reason given in Section 2.3.
- **Shorter excerpts in the beginner task:** 1,500 characters gave 0.984, and 1,000 characters 0.977, for 800 and 729 tokens per case.
- **More excerpts after a neutral answer** (advanced task, 150 cases): this fixed 2 misses but turned 3 correct neutral answers into contradictions, for 80 % more input tokens.
- **Four extra lines in the advanced prompt,** one for each of its main errors (who says what; arguments against a proposal do not contradict what it would do; compare every number; "not mentioned" is neutral): macro-F1 0.973 instead of 0.974, for 11 % more tokens. They fixed 15 errors and made 17 new ones, mostly neutral claims about another topic answered contradiction although the reasoning said the booklet does not mention them.
- **Three extra rules in the advanced prompt** (with the other edition): macro-F1 0.960 instead of 0.962. They traded contradictions answered neutral for neutrals answered contradiction.

**Tokens and time.** Narrowing the context cut the tokens per case 2.3-fold in the beginner task and 30-fold in the advanced task, while it raised macro-F1. The processing time leaves out Apertus' latency, which was 1.5 s per case in the beginner task and 1.9 s in the advanced task. That leaves 0.9 s per run of all 1,488 cases in the beginner task, and 5.4 s in the advanced task:

- **Beginner task:** all of it is start-up, mostly Python's imports, half of them the OpenAI client. The search runs while other cases' requests are in flight, so it adds nothing measurable.
- **Advanced task:** start-up takes 0.8 s. Reading the text layer of the 60 booklets before the first request takes 4.6 s, all of which counts. Reading their tags when a case first needs them takes about 17 s, but while other cases' requests are in flight, so practically none of it counts.

## 6. Limitations

- **No held-out split:** prompts and settings were chosen on the same 1,488 public cases we report, so the private benchmark may score lower.
- **Errors left:** in the beginner task, 5 of the errors that scoping introduced are contradictions whose deciding sentence was cut away. In the advanced task, most errors left are wrong answers of contradiction: 18 neutral claims and 10 entailments, most of the 10 same-language claims about what happens if the vote is accepted. Same-language claims are the weaker group (Table 2).
- **Scope** relies on the running headers. A booklet without them is searched whole.
- **No OCR:** all booklets are born-digital. A scanned booklet would have neither tags nor a text layer.
- **Time:** we measure in-process, on a laptop. The organisers time the container from outside, which also counts its start.

## 7. Reproducibility

- **Environment:** `uv sync` in `track_2a/`. All dependencies (`uv.lock`) and the dataset revision are pinned.
- **Data:** `uv run python -m src stage=prepare` downloads the dataset and the booklets.
- **Runs:** `uv run python -m src strategy=gied_beginner_scoped model=apertus-70b` (beginner task) and `uv run python -m src task=advanced strategy=gied_strict_reasoned model=apertus-70b` (advanced task). Each writes its predictions, scores, tokens and processing time to `data/runs/`.
- **The submission:** `make run` builds the Docker image and decides the sample cases. It needs only `API_KEY`, plus `BASE_URL` for another endpoint: no model weights and no GPU.
- **Hardware:** any CPU; all our runs were made from a laptop. Temperature 0, with no sampling.
- **Commit:** `exact commit`.

## 8. Next steps

- **Look into the wrong answers of contradiction** in the advanced task (18 neutral claims, 10 entailments), and into the same-language claims, which are now the weaker group.
- **Hide the rest of the advanced task's processing time:** read each booklet's text layer when a case first needs it, as the tags already are.
- **Validate on a held-out split,** and measure the cheaper 8B model.

## License

Creative Commons Attribution 4.0 (CC-BY-4.0). All HackApertus projects are open-sourced.

## References

- Artetxe, M., Goswami, V., Bhosale, S., Fan, A., Zettlemoyer, L. (2023). Revisiting Machine Translation for Cross-lingual Classification. _EMNLP 2023_.
- Auer, C., et al. (2024). Docling Technical Report. _arXiv:2408.09869_.
- Chen, T., et al. (2024). Dense X Retrieval: What Retrieval Granularity Should We Use? _EMNLP 2024_.
- Conneau, A., Rinott, R., Lample, G., Williams, A., Bowman, S. R., Schwenk, H., Stoyanov, V. (2018). XNLI: Evaluating Cross-lingual Sentence Representations. _EMNLP 2018_.
- Cormack, G. V., Clarke, C. L. A., Büttcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. _SIGIR 2009_.
- Dagan, I., Glickman, O., Magnini, B. (2006). The PASCAL Recognising Textual Entailment Challenge. _Machine Learning Challenges, LNCS 3944_.
- Egli, E., Mamié, N., Müller, M., Dolev, E. L. (2023). Voting Booklet Bias: Stance Detection in Swiss Federal Communication. _arXiv:2306.08999_.
- Etxaniz, J., Azkune, G., Soroa, A., Lopez de Lacalle, O., Artetxe, M. (2024). Do Multilingual Language Models Think Better in English? _NAACL 2024_.
- Gale, W. A., Church, K. W. (1993). A Program for Aligning Sentences in Bilingual Corpora. _Computational Linguistics 19(1)_.
- Guo, Z., Schlichtkrull, M., Vlachos, A. (2022). A Survey on Automated Fact-Checking. _TACL 10_.
- Hoyle, A., Sarkar, R., Goel, P., Resnik, P. (2023). Natural Language Decompositions of Implicit Content Enable Better Text Representations. _EMNLP 2023_.
- ISO 32000-2:2020, Portable Document Format, Part 2: PDF 2.0 (§14.8, Tagged PDF); ISO 14289-1:2014, PDF/UA-1.
- Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang, P. (2024). Lost in the Middle: How Language Models Use Long Contexts. _TACL 12_.
- McNamee, P., Nicholas, C., Mayfield, J. (2009). Addressing Morphological Variation in Alphabetic Languages. _SIGIR 2009_.
- Menick, J., et al. (2022). Teaching Language Models to Support Answers with Verified Quotes. _arXiv:2203.11147_.
- OST (2026). MNLIoverSwissVotingBooklets, v1.1. _huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets_.
- Project Apertus (2025). Apertus: Democratizing Open and Compliant LLMs for Global Language Environments. _arXiv:2509.14233_.
- Robertson, S., Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. _Foundations and Trends in Information Retrieval 3(4)_.
- Shi, F., Chen, X., Misra, K., Scales, N., Dohan, D., Chi, E., Schärli, N., Zhou, D. (2023). Large Language Models Can Be Easily Distracted by Irrelevant Context. _ICML 2023_.
- Thorne, J., Vlachos, A., Christodoulopoulos, C., Mittal, A. (2018). FEVER: a Large-scale Dataset for Fact Extraction and VERification. _NAACL 2018_.
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. _NeurIPS 2022_.
- Williams, A., Nangia, N., Bowman, S. R. (2018). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference. _NAACL 2018_.
- Xu, P., et al. (2024). Retrieval meets Long Context Large Language Models. _ICLR 2024_.
