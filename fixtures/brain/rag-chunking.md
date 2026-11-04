# Chunking Documents for Retrieval

## Size versus precision

Before a document can be searched, it is split into chunks, and the chunk is the unit that gets embedded and retrieved. Chunk size is a trade-off between context and precision. Large chunks carry more context, so an answer is less likely to be cut in half, but each one covers several ideas and its embedding becomes a blurry average of all of them, so precision drops. Small chunks are precise but may lose the sentence that explains what "it" refers to. A few hundred tokens per chunk is a common middle ground for prose.

## Overlap between windows

Use overlap between windows: each chunk repeats the last part of the previous one, for example fifty tokens. Without overlap, a key fact that straddles a boundary is split across two chunks and neither one matches the question well. Overlap costs a little extra storage and embedding time, and in exchange the facts near the edges stay findable.

## Keep section headings

Keep section headings with each chunk. Prefixing the chunk with its heading path, such as "Installation > Windows", tells both the embedder and the reader where the passage comes from. A chunk that says "run the installer again" is useless on its own and clear with its heading.
