# `part2/run_tests.py`

## Scopo

Esegue una grid search su un sottoinsieme di parametri per SAC o PPO, organizzata in 4 profili predefiniti. Ogni combinazione di parametri viene eseguita come processo separato e i log vengono salvati in `test/profile_N/<run_name>/training_log.txt`.

## Prerequisiti

Attivare il proprio ambiente conda prima di lanciare lo script:

```bash
conda activate <nome_env>
python run_tests.py --algorithm sac --profile 1
```

Lo script usa `sys.executable` (il Python dell'env attivo) per lanciare i sottoprocessi — non gestisce l'attivazione dell'env autonomamente.

## Argomenti

| Argomento | Valori | Obbligatorio | Descrizione |
|---|---|---|---|
| `--algorithm` | `sac`, `ppo` | Sì | Algoritmo da usare |
| `--profile` | `1`, `2`, `3`, `4` | Sì | Profilo di parametri da eseguire |
| `--smoke` | flag | No | Esegue ogni run con pochi timesteps per verificare che non crashino |

## Profili SAC

I profili SAC sono pensati per `PandaPush-v3` come configurato nel progetto:
reward dense, episodi da 500 step, osservazioni goal-conditioned e gap di massa
source/target 1kg/5kg. La ricerca parte quindi da SAC+HER e poi testa quanto
replay/update budget e domain randomization aiutano il transfer.

| Profilo | Parametri esplorati | Run totali |
|---|---|---|
| 1 | `learning_rate` × `gamma` × `batch_size` con `HER=True`, source fisso | 8 |
| 2 | `gamma` × `batch_size` × `gradient_steps` con `HER=True`, source fisso | 12 |
| 3 | `batch_size` × `gradient_steps` × `sampling_strategy` (`none`, `udr`, `adr`) con `HER=True` | 12 |
| 4 | shortlist per run lunghe: `learning_rate` × `batch_size` × `sampling_strategy` con `HER=True` | 8 |

## Profili PPO

PPO resta utile come baseline on-policy, ma in questo task i log esistenti
mostrano success rate bassi rispetto a quanto ci si aspetta da SAC+HER su un
task goal-conditioned. I profili PPO sono quindi piu' compatti e servono come
confronto.

| Profilo | Parametri esplorati | Run totali |
|---|---|---|
| 1 | `learning_rate` × `gamma`, rollout fisso e source fisso | 4 |
| 2 | `n_steps` × `clip_range` × `gae_lambda`, source fisso | 8 |
| 3 | `ent_coef` × `sampling_strategy` (`none`, `udr`, `adr`) | 6 |
| 4 | shortlist PPO: `learning_rate` × `sampling_strategy` | 4 |

## Struttura output

```
test/
└── profile_1/
    ├── SAC_1_lr1e-4_g0p98_bs256_gs1_ls1000_her1_drnone/
    │   └── training_log.txt
    ├── SAC_1_lr1e-4_g0p98_bs512_gs1_ls1000_her1_drnone/
    │   └── training_log.txt
    └── ...
```

Ogni `training_log.txt` contiene:
- Il comando eseguito
- Il timestamp di inizio
- L'output completo di stdout+stderr del processo

## Smoke test

Il flag `--smoke` riduce drasticamente i timesteps per verificare che ogni combinazione di parametri non produca errori:

| Algoritmo | Timesteps smoke |
|---|---|
| SAC | 2500 (supera `learning_starts=1000`, fa qualche update) |
| PPO | 3000 richiesti; SB3 completa almeno un rollout, quindi con `n_envs=8` puo' raccogliere piu' step |

```bash
# Verifica veloce che tutto funzioni
python run_tests.py --algorithm sac --profile 3 --smoke
python run_tests.py --algorithm ppo --profile 2 --smoke
```

## Esempi d'uso

```bash
# Smoke test su tutti i profili SAC
python run_tests.py --algorithm sac --profile 1 --smoke
python run_tests.py --algorithm sac --profile 2 --smoke
python run_tests.py --algorithm sac --profile 3 --smoke
python run_tests.py --algorithm sac --profile 4 --smoke

# Run completo profilo 3 SAC (HER + DR/ADR)
python run_tests.py --algorithm sac --profile 3

# Run completo profilo 4 PPO (shortlist)
python run_tests.py --algorithm ppo --profile 4
```

## Riepilogo finale

Al termine, lo script stampa un riepilogo con esito di ogni run:

```
============================================================
RIEPILOGO — 8 run
============================================================
  ✓  SAC_1_lr1e-4_g0p98_bs256_gs1_ls1000_her1_drnone
  ✓  SAC_1_lr1e-4_g0p98_bs512_gs1_ls1000_her1_drnone
  ✗  SAC_1_lr3e-4_g0p99_bs512_gs1_ls1000_her1_drnone
  ...

6/8 completate con successo.
```

Le run fallite hanno il log completo in `training_log.txt` per diagnosticare l'errore.

## Note

- Le run vengono eseguite sequenzialmente, non in parallelo — per parallelizzarle manualmente si può lanciare `run_tests.py` con profili diversi su terminali diversi
- Lo script chiama `train_sb3.py` (script unificato) — assicurarsi che sia presente nella stessa directory
- I log TensorBoard dello script unificato vengono salvati in `./tb_logs/`
