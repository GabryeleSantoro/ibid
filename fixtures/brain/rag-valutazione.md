# Valutare la ricerca

## Le metriche

Per capire se la ricerca funziona servono numeri, non impressioni. Le due metriche che uso di più sono recall@6 e MRR. La recall@6 misura in quante domande almeno un passaggio corretto compare tra i primi sei risultati, cioè tra quelli che finiscono davvero nel prompt. Il MRR, mean reciprocal rank, premia i risultati corretti messi in alto: vale 1 se il primo passaggio è giusto, 0,5 se è il secondo, e così via, mediato su tutte le domande.

## Il golden set

Le metriche si calcolano su un golden set di domande con i documenti attesi: un file con una domanda per riga e, accanto, il nome del file o della sezione che contiene la risposta. Le domande devono assomigliare a quelle vere degli utenti, in più lingue, e non devono copiare le parole esatte dei documenti, altrimenti la ricerca per parole chiave vince troppo facilmente.

## La baseline

Un numero da solo dice poco. Serve un confronto con una baseline: si salva il risultato di una versione buona e a ogni modifica si rilancia la valutazione. Se la recall scende oltre una soglia di tolleranza, il controllo automatico fallisce e la modifica va rivista prima di essere unita.
