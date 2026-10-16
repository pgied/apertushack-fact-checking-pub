# Technical report — Multilingual Natural Language Inference over Swiss Official Voting Booklets

- **Track:** Track 2A — OST: Multilingual Natural Language Inference over Swiss Official Voting Booklets
- **Event:** Online
- **Team:** FactCheckers — Fatma-Zohra Rezkellah, Sophia Conrad, Patrick Giedemann
- **Demo:** the Docker image `ghcr.io/pgied/apertushack-fact-checking-pub:latest` (no demo video required)
- **Code:** the beginner task is [`gied_beginner_scoped`](src/strategies/gied_beginner_scoped/strategy.py), the advanced task is [`gied_strict_reasoned`](src/strategies/gied_strict_reasoned/strategy.py); the shared retrieval code is in [`src/strategies/_gied/`](src/strategies/_gied/).

## 1. Summary

Every Swiss popular vote comes with an official booklet in German, French and Italian (and Romansh, not included in this study). The task is to decide whether the booklet supports a claim about the vote (entailment), contradicts it, or neither (neutral). Two thirds of the claims are in another language than the booklet. The beginner task gives a section of the booklet, the advanced task the whole PDF, and asks for the evidence too. Both our solutions retrieve, then verify (Thorne et al., 2018). Our code finds the few sentences that bear on the claim, and `Apertus-v1.5-70B` judges only those. On all 1,488 public cases, we reach a macro-F1 of 0.987 in the beginner task (baseline 0.788), with 834 tokens and 0.6 ms of processing time per case. In the advanced task we reach 0.974, with an evidence Hit@5 of 0.979, 1,157 tokens and 3.6 ms per case. Prompting with the whole booklet reached 0.70–0.73, for about 34,770 tokens per case.

## 2. Architecture

| Component                                         | What it does                                                        | Where it runs                           |
| ------------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------- |
| The CLI, [`src/entrypoint.py`](src/entrypoint.py) | Reads the cases, decides 8 at a time, writes one JSON line per case | Our Docker image: linux/amd64, CPU only |
| PDFium (`pypdfium2`)                              | Reads the booklet's text from its PDF tags                          | Our Docker image                        |
| BM25, our own Python code                         | Finds the sentences that bear on the claim                          | Our Docker image                        |
| Apertus v1.5 70B                                  | Translates a claim in another language, and decides the label       | A hosted endpoint, at `BASE_URL`        |

There is no OCR, no embedding model and no other model. A case sends one request to Apertus, or two if the claim is in another language than the document: a translation, then the decision. In Figures 1 and 2, the numbers are the steps of Sections 2.2 and 2.3, and a dashed box is a step that only some cases take.

### 2.1 Shared by both tasks: sentence search and translation

- **Sentences** are the unit of search, since the evidence has to be short and verbatim. Chen et al. (2024) found, with dense retrievers, that sentences fit more of the relevant text into a fixed number of tokens than passages, and propositions more still. But propositions are rewritten by a model, so they could not be quoted. A sentence never crosses a paragraph or a page, an abbreviation such as "z. B." does not end it, and one under 40 characters joins the next.
- **BM25** ranks the sentences for the claim (k1 = 1.2, b = 0.75; Robertson and Zaragoza, 2009). Words are lower-cased, stripped of accents and cut to their first 7 letters. This is a language-independent stand-in for stemming (McNamee et al., 2009), which found more evidence than 5, 6 or whole words. A sentence also gets 30 % of its better neighbour's score.
- **Translation:** BM25 needs the claim in the document's language, so Apertus first translates a claim in another language. This is translate-test (Conneau et al., 2018; Artetxe et al., 2023), with the model translating for itself (self-translate; Etxaniz et al., 2024).

### 2.2 Beginner task: a scoped reference

**Figure 1: Beginner task**

![The beginner task: translate the claim, scope the reference, decide](docs/architecture_beginner.svg)

1. **Translate** a claim into the reference's language if it is not the same.
2. **Scope** a reference over 2,000 characters, as 1,225 of the 1,488 are: the 6 sentences that match the claim best, each with the sentence before and after it, in order, with "[…]" for each gap.
3. **Decide:** Apertus explains in at most three sentences what the reference says about the claim, then writes `Label: <word>` (Section 3).

The evidence is the reference shown, with page `null`; a neutral answer gets none.

### 2.3 Advanced task: retrieve, then verify

**Figure 2: Advanced task**

![The advanced task: read and index the booklet once, then narrow to the claim's vote, search, build the excerpts, decide and return the evidence](docs/architecture_advanced.svg)

1. **Read the booklet** once, when a case first needs it, from its PDF tags (ISO 32000-2 §14.8; PDF/UA, ISO 14289-1): a structure tree of headings and paragraphs in reading order, which leaves out running headers and page numbers. The text layer stands in where tags are missing: for the untagged voting text, and for the one booklet without tags (French, 28 November 2021).
2. **Index its sentences,** each with its page and the two headings above it.
3. **Narrow to the claim's vote.** Running headers such as "Erste Vorlage: …" assign the pages to the votes, and we keep the vote whose pages share the most rare words and word pairs with the case's vote title. That is a third of a booklet on average, with all gold pages in 989 of 990 cases.
4. **Search** these pages with the claim, or with its translation.
5. **Build the excerpts:** the 8 best sentences, each with its neighbours on the page, about 2,000 characters, numbered and labelled with page and section, e.g. `[3] (p. 12, Argumente › Zu strikte Vorgaben)`.
6. **Decide** as in the beginner task, and name the deciding excerpts: `Excerpts: <numbers>`.
7. **Return the evidence:** five verbatim sentences with their pages, those of the cited excerpts first, but never the vote's title, which starts every gold passage. A neutral answer gets evidence too, since the evidence of every gold entailment or contradiction is scored.

| Choice                                                     | Why                                                                                                                                                                                                                                         |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| About 2,000 characters of context                          | References have up to 25,000 characters, booklets 23,000 to 270,000. Long and distracting contexts lower accuracy (Liu et al., 2024; Shi et al., 2023), and short retrieved contexts can match long-context models (Xu et al., 2024).       |
| Only the claim's vote                                      | Claims about the voting text or the summary read alike for every vote, so they had often been judged against another vote's text.                                                                                                           |
| The PDF tags                                               | They kept 97.9 % of each gold passage, at 3 ms per page; the text layer 96.0 % at 1 ms, and Docling (Auer et al., 2024) 95.9 % at 157 ms.                                                                                                   |
| Only the supplied booklet                                  | The challenge asks to judge against the supplied source. Also searching its edition in the claim's language raised macro-F1 from 0.951 to 0.962, but that edition is in the input only when another case uses it.                           |
| Neutral first, contradiction only with a quotable sentence | Most of the baseline's errors were neutral claims answered contradiction (Section 5). As in answers backed by quotes (Menick et al., 2022), a contradiction has to rest on a sentence the model can quote; the quote itself is not checked. |
| Reasoning before the label                                 | Reasoning first helps (Wei et al., 2022), and an explanation written after the label cannot change it.                                                                                                                                      |

## 3. Use of Apertus

- **Model:** `swiss-ai/Apertus-v1.5-70B` (Project Apertus, 2025), inference only, prompted zero-shot at temperature 0. The 8B model served during development and scored lower.
- **Calls:** the translation of a cross-lingual claim (at most 200 output tokens), and the decision: at most three sentences of reasoning, then the label (at most 200 output tokens; 250 in the advanced task, whose answer also names the excerpts). Only the last `Label:` and `Excerpts:` lines are read, since the reasoning may name other labels and numbers.
- **Endpoint:** the OpenAI-compatible chat API at `BASE_URL`, the CSCS Inference API during development and the organisers' proxy at the evaluation, with 8 requests in parallel.
- **Prompt:** the label definitions below, then the claim with its translation. In the advanced task it says "booklet" for "reference", gives the vote and the numbered excerpts, and says that only the booklet counts, not the model's own knowledge.

> **neutral:** the reference neither supports nor contradicts the claim<br>
> **contradiction:** the reference says the opposite of the claim, and you can quote the sentence that does<br>
> **entailment:** the reference supports the claim<br>
> If the reference does not mention what the claim says, or is about another topic, the label is "neutral", not "contradiction": a claim the reference does not support is not contradicted by it.

## 4. Data

- **Dataset:** `OSTswiss/MNLIoverSwissVotingBooklets`, version v1.1, pinned to one revision (OST, 2026), MIT licence. Its 1,488 rows each give a beginner and an advanced case with the same claim and label. The labels are balanced, the claims split evenly between German, French and Italian, and two thirds of the cases are cross-lingual.
- **Booklets:** the 60 official booklets the dataset refers to, from the Federal Chancellery (bk.admin.ch). They are public and not redistributed, except one for the `make run` sample. There is no personal data. Swiss voting booklets have also been studied for stance detection (Egli et al., 2023).

## 5. Evaluation

We report the challenge's metrics on all 1,488 public cases: macro-F1; evidence Hit@5, whether one of the first five evidence items matches the gold passage (rapidfuzz `partial_ratio` ≥ 90, as in the challenge's `evaluate.py`); tokens per case, as the API reports them; and processing time per case, without Apertus' latency ([`src/timing.py`](src/timing.py)). All runs used Apertus v1.5 70B on the CSCS Inference API. Repeated identical requests changed 4 of 515 labels, so differences under about 0.005 are noise.

**Table 1: Results** (all 1,488 cases, unless noted; ↑ higher is better, ↓ lower is better). Time is the processing time per case, without Apertus' latency.

| Task     | System                                                                   | Macro-F1 ↑    | Hit@5 ↑   | Tokens per case (in + out) ↓ | Time ↓ |
| -------- | ------------------------------------------------------------------------ | ------------- | --------- | ---------------------------- | ------ |
| Beginner | Baseline: the whole reference, a one-word answer                         | 0.788         | –         | 1,897 + ≤ 10                 | –      |
| Beginner | **Our submitted model**                                                  | **0.987**     | –         | 748 + 86                     | 0.6 ms |
| Advanced | Baseline: full-document prompting, the whole booklet (150 cases, 2 runs) | 0.701 / 0.727 | 0.128     | 34,762 + ≤ 10                | –      |
| Advanced | Ours, on the same 150 cases                                              | 0.978         | 1.000     | 1,048 + 102                  | –      |
| Advanced | **Our submitted model**                                                  | **0.974**     | **0.979** | 1,053 + 104                  | 3.6 ms |

**Table 2: What each step added** (all 1,488 cases, unless noted)

| Task     | Step                                                                                         | Macro-F1 ↑ | Hit@5 ↑   | Input tokens per case ↓ |
| -------- | -------------------------------------------------------------------------------------------- | ---------- | --------- | ----------------------- |
| Beginner | Baseline prompt                                                                              | 0.788      | –         | 1,897                   |
| Beginner | Neutral first, reasoning before the label                                                    | 0.901      | –         | 1,952                   |
| Beginner | `quote` definitions                                                                          | 0.962      | –         | 1,984                   |
| Beginner | + the translation of a cross-lingual claim                                                   | 0.965      | –         | 2,092                   |
| Beginner | **+ only the sentences around the claim (ours)**                                             | **0.987**  | –         | **748**                 |
| Advanced | Full-document prompting (150 cases)                                                          | 0.70–0.73  | 0.13      | 34,762                  |
| Advanced | Only the vote's pages; the model's quotes are looked up in the PDF                           | 0.850      | 0.485     | 11,553                  |
| Advanced | Retrieve, then verify: BM25 over the vote's pages with the translated claim, the label first | 0.951      | 0.978     | 1,077                   |
| Advanced | **+ reasoning before the label (ours)**                                                      | **0.974**  | **0.979** | **1,053**               |

**Languages.** Cross-lingual cases do as well as same-language ones: macro-F1 0.985 against 0.990 in the beginner task, and 0.982 against 0.958 in the advanced task. By the language of the claim or of the document, macro-F1 stays between 0.980 and 0.990 (beginner) and between 0.963 and 0.987 (advanced). The weakest pair is German claims on German booklets, 0.923 in the advanced task.

**What made the difference** was the line between neutral and contradiction. In the beginner task, 274 of the baseline's 498 errors were neutral claims answered contradiction; putting neutral first with reasoning cut them to 140, and the `quote` definitions to 46. Scoping fixed 44 of the 52 errors left, since irrelevant context distracts language models (Shi et al., 2023), and made 12 new ones. In the advanced task, answering with the label first missed 42 contradictions, also with 250 tokens instead of 40. Reasoning first left 4.

**Tried and left out:**

- **Verified quotes,** our first advanced approach: Apertus got only the vote's pages and answered in JSON with the label and up to three sentences copied from the booklet, which we looked up in the PDF, exactly or as the closest passage, to give the evidence with its real page, after answers backed by verified quotes (Menick et al., 2022). It reached 0.850 and a Hit@5 of 0.485, for 11,553 input tokens per case. Asking for quotes on neutral answers too raised Hit@5 to 0.642 but lowered macro-F1 to 0.836. Retrieving the sentences ourselves cut the tokens about tenfold and makes the evidence verbatim by construction.
- **Paraphrases** of the claim in the search (Hoyle et al., 2023), merged by reciprocal rank fusion (Cormack et al., 2009): 0.979 in the beginner task, and 0.947 instead of 0.951 in the advanced task, for three times and 1.25 times the tokens. They turned more neutral claims into contradictions, mostly claims about another vote whose explanation said the booklet does not mention them. Reading such answers as neutral removed nearly all of these, but also some true contradictions explained the same way ("the booklet does not mention five years, it says ten").
- **Also searching the booklet's edition in the claim's language,** each sentence found mapped to the same position on the same page (Gale and Church, 1993): 0.962 instead of 0.951, left out for the reason in Section 2.3.
- **Less context in the beginner task,** 1,500 or 1,000 characters: 0.984 and 0.977, for 800 and 729 tokens per case instead of 834.
- **Four prompt rules** against the main advanced errors: 0.973, for 11 % more tokens. They fixed 15 errors and made 17 new ones.

**Tokens and time.** Narrowing the context cut the tokens per case 2.3-fold in the beginner task and 30-fold in the advanced task, while it raised macro-F1. Without Apertus' latency (1.5 s and 1.9 s per case), a run of all 1,488 cases takes 0.9 s in the beginner task, all of it start-up, and 5.4 s in the advanced task, 4.6 s of it reading the booklets' text layers before the first request. Their tags are read while requests are in flight.

## 6. Limitations

- **Prompt engineering only:** besides the paraphrases, we did not test more complex methods.
- **Errors left:** most are wrong contradictions in the advanced task: 18 neutral claims, and 10 entailments, mostly same-language claims about what happens if the vote is accepted. In the beginner task, scoping cut away the deciding sentence of 5 contradictions.
- **Booklet structure:** narrowing to the vote relies on running headers. A booklet without them is searched whole. The booklet without tags is read without subheadings; 14 of its 17 cases are right, and the 3 wrong are one repeated claim.
- **No OCR:** a scanned booklet would have neither tags nor a text layer.
- **Time** is measured in-process on a laptop. The organisers time the whole container, including its start.
- **Task difficulty:** the task would be closer to reality if the correct booklet were not given but had to be found by Apertus itself. Since some voting questions come back over time, such as Switzerland joining the EU, this could make the task considerably harder.

## 7. Reproducibility

- **Setup:** `uv sync` in `track_2a/`, then `uv run python -m src stage=prepare` for the dataset and the booklets. Dependencies (`uv.lock`) and the dataset revision are pinned.
- **Runs:** `uv run python -m src strategy=gied_beginner_scoped model=apertus-70b` and `uv run python -m src task=advanced strategy=gied_strict_reasoned model=apertus-70b`, with their results in `data/runs/`.
- **Submission:** `make run` builds the Docker image and decides the sample cases. It needs `API_KEY` (and `BASE_URL` for another endpoint), any CPU, no GPU and no model weights.

## 8. Next steps

- **The search, more than the prompt.** With the source given, the task is largely one of prompt refinement. The label definitions and reasoning first alone raised beginner macro-F1 from 0.788 to 0.962 (Table 2). Fact verification proper starts by finding the source (Thorne et al., 2018): which of the 60 booklets, which edition and which vote. That is not trivial, since topics such as immigration or the relationship with the EU come back to the vote over the years.
- **Above 0.9, the labels are open to interpretation.** On MNLI, the majority of five annotators matches the gold label in only about 92 % of the cases (Nangia and Bowman, 2019), and such disagreement is often genuine (Pavlick and Kwiatkowski, 2019). Our runs show it too: four extra prompt rules fixed 15 errors and made 17 new ones. Several people relabelling our errors would tell mistakes from disagreements.
- **Smaller steps:** test the cheaper 8B model, and read each booklet's text layer only when a case first needs it.

## License

Creative Commons Attribution 4.0 (CC-BY-4.0). All HackApertus projects are open-sourced.

## References

- Artetxe, M., et al. (2023). Revisiting Machine Translation for Cross-lingual Classification. _EMNLP 2023_.
- Auer, C., et al. (2024). Docling Technical Report. _arXiv:2408.09869_.
- Chen, T., et al. (2024). Dense X Retrieval: What Retrieval Granularity Should We Use? _EMNLP 2024_.
- Conneau, A., et al. (2018). XNLI: Evaluating Cross-lingual Sentence Representations. _EMNLP 2018_.
- Cormack, G. V., Clarke, C. L. A., Büttcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. _SIGIR 2009_.
- Egli, E., et al. (2023). Voting Booklet Bias: Stance Detection in Swiss Federal Communication. _arXiv:2306.08999_.
- Etxaniz, J., et al. (2024). Do Multilingual Language Models Think Better in English? _NAACL 2024_.
- Gale, W. A., Church, K. W. (1993). A Program for Aligning Sentences in Bilingual Corpora. _Computational Linguistics 19(1)_.
- Hoyle, A., et al. (2023). Natural Language Decompositions of Implicit Content Enable Better Text Representations. _EMNLP 2023_.
- ISO 32000-2:2020, Portable Document Format, Part 2: PDF 2.0 (§14.8, Tagged PDF); ISO 14289-1:2014, PDF/UA-1.
- Liu, N. F., et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. _TACL 12_.
- McNamee, P., Nicholas, C., Mayfield, J. (2009). Addressing Morphological Variation in Alphabetic Languages. _SIGIR 2009_.
- Menick, J., et al. (2022). Teaching Language Models to Support Answers with Verified Quotes. _arXiv:2203.11147_.
- Nangia, N., Bowman, S. R. (2019). Human vs. Muppet: A Conservative Estimate of Human Performance on the GLUE Benchmark. _ACL 2019_.
- OST (2026). MNLI over Swiss Voting Booklets: Benchmark Dataset, v1.1. _huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets_.
- Pavlick, E., Kwiatkowski, T. (2019). Inherent Disagreements in Human Textual Inferences. _TACL 7_.
- Project Apertus (2025). Apertus: Democratizing Open and Compliant LLMs for Global Language Environments. _arXiv:2509.14233_.
- Robertson, S., Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. _Foundations and Trends in Information Retrieval 3(4)_.
- Shi, F., et al. (2023). Large Language Models Can Be Easily Distracted by Irrelevant Context. _ICML 2023_.
- Thorne, J., et al. (2018). FEVER: a Large-scale Dataset for Fact Extraction and VERification. _NAACL 2018_.
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. _NeurIPS 2022_.
- Xu, P., et al. (2024). Retrieval meets Long Context Large Language Models. _ICLR 2024_.
