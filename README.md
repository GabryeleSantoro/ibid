<div align="center">

<img src="apps/desktop/src-tauri/icons/icon.svg" alt="Ibid" width="120" />

# Ibid

**A librarian for your files.**<br>
Ask questions about your documents and get answers that cite the exact page.

[![Latest release](https://img.shields.io/github/v/release/GabryeleSantoro/ibid?style=for-the-badge&color=3a6cf4&label=release)](https://github.com/GabryeleSantoro/ibid/releases/latest)
[![Platform](https://img.shields.io/badge/macOS-Apple%20Silicon%20%26%20Intel-0f1d4d?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/GabryeleSantoro/ibid/releases/latest)
[![Local first](https://img.shields.io/badge/local--first-your%20files%20stay%20put-f5a524?style=for-the-badge)](#privacy-in-practice)
[![Cited answers](https://img.shields.io/badge/answers-always%20cited-9db7ff?style=for-the-badge)](#chat)

</div>

---

Point Ibid at folders of documents, then ask questions in plain language. It finds the relevant passages, writes an answer from them, and links every claim back to the page it came from.

- **Private.** Reading, indexing and search all happen on your computer. Only your question and the retrieved passages reach the model that writes the answer, and that model can run locally too.
- **Verifiable.** Every claim carries a citation. One click opens the document at the exact spot. If nothing relevant is found, Ibid says so instead of making something up.
- **Non-invasive.** Your files stay where they are. Ibid never moves, copies or modifies them.
- **Your choice of model.** Use OpenRouter, OpenAI, Anthropic, or a local model through LM Studio, Ollama or any OpenAI-compatible server.

Supported formats: **PDF**, **PowerPoint (PPTX)**, **Markdown** and **plain text**.

## Contents

[Installation](#installation) · [System requirements](#system-requirements) · [First launch](#first-launch) · [Using Ibid](#using-ibid) · [Privacy in practice](#privacy-in-practice) · [FAQ](#faq) · [Issues and feedback](#issues-and-feedback)

## Installation

1. Download the latest version from the [Releases](https://github.com/GabryeleSantoro/ibid/releases/latest) page.
2. Open the downloaded file and move Ibid to your Applications folder.
3. Launch it.

Ibid checks for updates on its own and can install them in the background (Settings → Updates). You can turn that off or check manually.

## System requirements

Ibid runs on macOS only. Search runs on your machine, so memory and disk matter more than the processor.

| | Minimum | Recommended |
| --- | --- | --- |
| **Mac** | Intel or Apple Silicon | Apple Silicon (M1 or later) |
| **Memory** | 8 GB | 16 GB or more |
| **Free disk** | 3 GB | 10 GB or more, plus room for the index of your library |
| **Answer model** | A remote connection (OpenRouter, OpenAI, Anthropic) | Remote, or the built-in local model on 16 GB+ |
| **Network** | Needed once to download models, and for remote connections | Same |

What the figures are based on:

- **Downloads.** Embedder about 0.3 GB, reranker about 0.6 GB. The optional built-in answer model is about 1.1 GB.
- **Memory while working.** The built-in answer model takes about 2.1 GB while it writes an answer and is released after 60 idle seconds. The embedder and reranker stay loaded alongside the app.
- **Speed.** On Apple Silicon, Ibid uses the GPU through Metal and shares unified memory with it. On an Intel Mac everything runs on the CPU, so indexing and local answers are noticeably slower; a remote connection is the better choice there.
- **Index size.** It grows with your library. It lives in `~/.ibid`, so keep free space on your home volume.

The 2.1 GB figure was measured on an Apple Silicon Mac. The other memory and disk limits are conservative estimates, not benchmarks, and Ibid has not been tested on older macOS releases. Settings → Performance and Diagnostics show what your Mac was detected as.

## First launch

A four-step guide prepares everything. You can skip any step and finish it later.

| Step | What happens |
| --- | --- |
| **1. Your computer** | Ibid detects your CPU, memory and GPU and picks matching defaults. You can change them later in Settings → Performance. |
| **2. Base models** | Two small local models are downloaded: one turns text into searchable vectors (the *embedder*), the other re-orders the best candidates by relevance (the *reranker*). Downloads resume if interrupted. |
| **3. Model for answers** | Add a *connection*: pick a service, paste your API key, activate it. This model only turns the retrieved passages into a written answer; search itself stays local. Keys are stored in your system keychain. |
| **4. First folder** | Choose the folder holding your documents. Ibid starts indexing it right away and shows progress. |

## Using Ibid

### Chat

Type a question and press Enter (Shift+Enter adds a line). You see the search step first, then the answer streams in.

- **Citations.** Each claim is followed by a citation. Click it to open the Reader on the source page with the passage highlighted.
- **Trust signals.** If an answer has no citations, or rests on a single passage, the chat tells you. A citation that doesn't point at a passage Ibid actually retrieved is removed before you see it.
- **Passages panel.** See which passages were sent to the model, which were cited and which were retrieved but not used.
- **Scope.** Search your whole library, or narrow a chat to specific files or folders.
- **Projects.** Group related chats into projects, each with its own set of files to search.
- **Starter questions.** An empty chat suggests questions written in the language of your documents. Optionally, a small local model (about 1.1 GB) can write better ones.
- **Remote model warning.** With a remote connection, the chat reminds you that retrieved passages are sent to that provider. Embedding and reranking always stay local.

### Library

Every document and its indexing status in one place.

- Add folders as *sources*. Turn on reindexing on change to keep the index fresh automatically, or press **Reindex all**.
- Filter by title or path, and organize documents into your own folders. This doesn't move anything on disk.
- If a file fails to index, the Library shows the reason.
- Removing a source removes it from Ibid's index only. Your files are untouched.

### Reader

Opens any document, and is where citations land. Use **Ask about this document** to start a chat scoped to just that file.

### Slides → text

Turns slides into a complete Markdown document, with web sources listed at the end.

1. Select one or more indexed slide decks.
2. Optionally say what to go deeper on (for example, proofs or numerical examples).
3. Choose a title and the output language, then generate.

The result is saved as a Markdown file in your global folder. Requires an active connection.

### Models

See the embedder and reranker in use, and swap them. Browse Hugging Face for models in GGUF format; a fit badge shows whether each one will run well on your hardware. Changing the embedder makes existing vectors meaningless, so Ibid asks you to confirm and then re-indexes everything.

### Settings

- **General:** language of the interface and of the answers, and appearance.
- **Connections:** add, edit, activate and remove the models that write answers.
- **Retrieval:** how many passages are sent to the model, the minimum relevance score below which a passage is discarded (this is what allows a "not found" answer), and context budgets.
- **Performance:** batch sizes and GPU offload, preset from your hardware.
- **Storage:** where data lives, crash reports (opt-in, never including document content), and erasing the index.
- **Updates:** current version, automatic updates and manual check.

### Diagnostics and Logs

Diagnostics shows what is running, what has been indexed and your hardware. **Copy debug report** produces a report with no API keys or session tokens. Logs shows everything the app recorded this session, with a filter for errors and warnings.

## Privacy in practice

| Stays on your computer | Sent to your chosen model |
| --- | --- |
| Your files and their full text | Your question |
| The index and all embeddings | The few passages retrieved for it |
| Embedding and reranking | Recent chat history, for context |
| API keys (system keychain) | |

With a local model, nothing leaves your machine. Data lives in `~/.ibid`; Settings → Storage can erase the index at any time without touching your files.

## FAQ

**Do my documents leave my computer?**
No. They are read, indexed and searched locally. Only your question and the few passages needed to answer it are sent to the model you chose. If that model is local, nothing leaves your machine; with a remote one, the chat warns you.

**Which model should I connect?**
Any you already have an account for. OpenRouter gives access to many models with one key; a local model through LM Studio or Ollama keeps everything offline.

**Does it cost anything to run?**
Remote providers bill you for the tokens they use. Local models cost nothing beyond your hardware.

**Chat answers look generic or canned.**
No connection is active yet. Add and activate one in Settings → Connections.

**The answer says it couldn't find anything.**
That is intended: when nothing relevant clears the minimum score, Ibid declines instead of guessing. Try rephrasing, check the document is indexed in the Library, or lower the minimum score in Settings → Retrieval.

**A document isn't showing up.**
Open the Library: each file shows its status and, if indexing failed, the reason. Very large files may exceed the maximum file size set for the source.

**I changed a file. Do I need to do anything?**
Enable reindexing on change for the source, or press **Reindex all** in the Library.

**Will removing a folder delete my files?**
No. It only removes them from Ibid's index.

**I switched the embedder and search stopped working.**
Vectors from a different embedder are incompatible. Re-index your documents and search works again.

**Can I use it in another language?**
Yes. Change the language in Settings → General; it applies to the interface and to the answers.

**Is there a Windows or Linux version?**
Not yet. Releases are macOS builds for Apple Silicon and Intel.

**Where is my data stored?**
In `~/.ibid`. API keys are in your system keychain. Settings → Storage can erase the index.

**How do I uninstall it?**
Delete the app from Applications. To remove all data, first use Settings → Storage → Erase index, then delete `~/.ibid`.

## Issues and feedback

Found a bug or have an idea? [Open an issue](https://github.com/GabryeleSantoro/ibid/issues/new).

For bugs, please include:

- what you did and what you expected to happen;
- your Ibid version (Settings → Updates);
- the report from **Diagnostics → Copy debug report**.

Before opening a new one, take a look at the [existing issues](https://github.com/GabryeleSantoro/ibid/issues): someone may have reported it already.
