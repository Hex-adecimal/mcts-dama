# 📂 Guida ai Dati e Log: MCTS Dama

Questo documento spiega l'organizzazione delle cartelle generate durante il ciclo di Self-Play e Training.

## 🗄️ Dati (`data/`)

Questa cartella contiene tutti i dataset (partite) generati e processati.

| Percorso | Descrizione |
| :--- | :--- |
| **`data/bootstrap/`** | Contiene i dataset iniziali (euristici) usati per il "Bootstrap" della rete prima del self-play. |
| **`data/selfplay/`** | Area di lavoro temporanea per la generazione corrente. |
| `data/selfplay/raw_gen_N.bin` | File parziali generati dai singoli processi paralleli. Vengono sovrascritti ad ogni iterazione. |
| **`data/selfplay/new_data.bin`** | **Il file più importante.** È il risultato del merge dell'iterazione corrente. Il trainer legge da qui. |
| **`data/iteration/`** | **Archivio storico.** Contiene una copia di backup dei dati di ogni iterazione completata. |
| `data/iteration/iter_X_data.bin` | Dataset completo generato durante l'iterazione X. Utile per ri-addestrare da zero o analizzare la storia. |

---

## 🧠 Modelli e Pesi (`models/`)

Qui risiede l'intelligenza dell'agente.

| Percorso | Descrizione |
| :--- | :--- |
| **`models/cnn_weights.bin`** | **La rete attiva.** Contiene sempre i pesi più recenti e forti. È il file caricato dal gioco e dal trainer. |
| `models/cnn_weights_bootstrap.bin` | Backup dei pesi iniziali (pre-selfplay) per sicurezza. |
| **`models/archive/`** | **Storia evolutiva.** Copia dei pesi alla fine di ogni iterazione. |
| `models/archive/iter_X_weights.bin` | I pesi della rete al termine dell'iterazione X (usata poi per generare dati per la X+1). |

---

## 📝 Log (`logs/`) - *Da Implementare nella struttura finale*

Se lo script `train_loop.sh` è configurato per salvare i log (attualmente stampa a video o redireziona):

| Percorso | Descrizione |
| :--- | :--- |
| `logs/selfplay/` | Output dei processi di generazione (controlla errori o velocità di generazione). |
| `logs/training/` | Output del trainer (Loss, Accuracy, Validation). Utile per graficare l'apprendimento. |

---

## 🔄 Il Flusso del Loop (`train_loop.sh`)

Quando esegui `./scripts/train_loop.sh`:

1. **Generate:** I thread scrivono in `data/selfplay/raw_*`, poi uniti in `data/selfplay/new_data.bin`.
2. **Archive Data:** `new_data.bin` viene copiato in `data/iteration/iter_N_data.bin`.
3. **Backup Weights:** `models/cnn_weights.bin` (stato corrente) viene copiato in `models/archive/iter_N_weights.bin`.
4. **Train:** Il trainer legge `new_data.bin` e AGGIORNA `models/cnn_weights.bin` (sovrascrivendolo con la versione migliorata).
5. **Repeat:** La nuova `cnn_weights.bin` viene usata per generare i dati dell'iterazione successiva.
