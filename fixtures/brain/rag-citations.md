# Citations in Generated Answers

## Cite with id and page

An answer generated from retrieved passages should show where each claim came from. Ask the model to cite with document id and page, using a fixed marker format such as [doc_123:4] placed right after the claim it supports. Each passage in the prompt carries a header with exactly that id and page, so the model can copy it rather than invent one. The interface turns each marker into a link that opens the document at the right page.

## Validate every marker

Models sometimes produce citations that look right but are not: a document id that was never retrieved, or a page that is not in the passage. Drop citations no retrieved passage backs. After the answer is generated, parse every marker and check it against the list of passages that were actually in the prompt. Anything that does not match is removed and counted.

## Why it matters

Citation checking keeps the answer honest. A user who sees a citation trusts that clicking it will show the evidence. One fabricated reference erodes that trust for every other answer. Reporting how many markers were dropped also signals when the model is straying from its sources, which is a useful warning in its own right.
