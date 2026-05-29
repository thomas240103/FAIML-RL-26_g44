# `part2/rand_wrapper.py`

## Scopo

Questo file definisce `RandomizationWrapper`, un wrapper Gym che modifica parametri fisici del task `PandaPush-v3` a ogni `reset()`.

L'obiettivo e' simulare tre scenari:

- `none`: nessuna randomizzazione
- `udr`: Uniform Domain Randomization
- `adr`: Adaptive Domain Randomization

## Perche' serve

Nel progetto, il dominio `source` usa massa `1.0` e il dominio `target` usa massa `5.0`. Se alleniamo solo su una fisica fissa, il transfer puo' essere debole. La randomizzazione serve a rendere la policy piu' robusta rispetto a massa e attrito.

## Modifiche introdotte

La versione iniziale del file era solo uno scheletro con `TODO`. Sono state aggiunte le seguenti parti:

- import di `deque` e `numpy`
- validazione del parametro `mode`
- memorizzazione dei limiti globali della massa
- lettura della massa nominale dell'ambiente
- gestione del range corrente `mass_min`, `mass_max`
- implementazione di `_sample_mass()`
- implementazione di `_sample_friction()`
- implementazione di una logica ADR semplice
- aggiornamento del range ADR in `step()`
- cambio reale della massa fisica in `reset()` tramite `changeDynamics(...)`
- cambio dell'attrito fisico in `reset()` tramite `set_lateral_friction(...)` e `set_spinning_friction(...)`

## Logica del costruttore

Nel costruttore vengono inizializzate queste variabili:

- `mass_min_limit`, `mass_max_limit`: limiti assoluti del range di randomizzazione
- `object_lateral_friction_min/max`: range dell'attrito laterale dell'oggetto
- `table_lateral_friction_min/max`: range dell'attrito laterale del tavolo
- `object_spinning_friction_min/max`: range dell'attrito rotazionale dell'oggetto
- `nominal_mass`: massa di default dell'ambiente corrente
- attriti nominali letti dalla simulazione PyBullet
- `mass_min`, `mass_max`: range usato per il sampling attuale
- `success_history`: finestra scorrevole dei successi usata da ADR
- `adr_high_threshold`, `adr_low_threshold`, `adr_step`: iperparametri di adattamento

Se `mode == "udr"`, il range della massa viene subito impostato all'intervallo completo e gli attriti vengono campionati uniformemente a ogni episodio.

Se `mode == "adr"`, il range parte dalla sola massa nominale e si allarga solo quando la policy si comporta bene. Per ora ADR adatta solo la massa; l'attrito resta nominale.

## `_sample_mass()`

Questa funzione decide quale massa usare al prossimo `reset()`.

- `none`
  - non cambia la massa
  - ritorna `None`
- `udr`
  - campiona uniformemente tra `mass_min_limit` e `mass_max_limit`
- `adr`
  - campiona uniformemente nel range adattivo corrente `mass_min ... mass_max`

## `_sample_friction()`

Questa funzione decide quali attriti usare al prossimo `reset()`.

- `none`
  - non cambia gli attriti nominali
- `udr`
  - campiona uniformemente:
    - attrito laterale dell'oggetto
    - attrito laterale del tavolo
    - attrito rotazionale dell'oggetto
- `adr`
  - per ora non randomizza gli attriti

I default di training sono volutamente piu' conservativi dell'evaluation hard, per evitare di valutare esattamente sulla stessa distribuzione usata da UDR:

- `object_lateral_friction_range=(0.5, 1.2)`
- `table_lateral_friction_range=(0.5, 1.2)`
- `object_spinning_friction_range=(0.0, 0.005)`

L'evaluation hard puo' invece usare range piu' ampi, ad esempio `0.2-1.5` per gli attriti laterali e `0.0-0.01` per l'attrito rotazionale.

## `_update_adr_range()`

Questa funzione aggiorna il range ADR in base al successo medio recente.

Regola implementata:

- se il successo medio e' alto, il range si allarga
- se il successo medio e' basso, il range si restringe verso la massa nominale

Piu' nel dettaglio:

- se `success_rate >= 0.8`
  - `mass_min` diminuisce
  - `mass_max` aumenta
- se `success_rate <= 0.2`
  - il range si richiude verso `nominal_mass`

Questa non e' una ADR avanzata da paper, ma e' una versione semplice, stabile e adatta a un progetto didattico.

## `step()`

Lo `step()` chiama normalmente `self.env.step(action)` e, se la modalita' e' `adr` e l'episodio e' terminato:

- legge `info["is_success"]` se disponibile
- aggiunge il risultato alla finestra dei successi
- prova ad aggiornare il range ADR

Questo significa che l'adattamento avviene una volta per episodio, non a ogni singolo timestep.

## `reset()`

In `reset()` viene:

1. campionata una nuova massa con `_sample_mass()`
2. campionata una nuova configurazione di attrito con `_sample_friction()`
3. modificata la dinamica fisica dell'oggetto se la massa non e' `None`
4. modificati gli attriti se la modalita' e' `udr`
5. eseguito il `reset()` normale dell'ambiente

La modifica fisica della massa avviene accedendo a:

- `self.env.unwrapped.task.sim`
- `sim._bodies_idx["object"]`
- `sim.physics_client.changeDynamics(...)`

La modifica degli attriti usa:

- `sim.set_lateral_friction("object", -1, ...)`
- `sim.set_lateral_friction("table", -1, ...)`
- `sim.set_spinning_friction("object", -1, ...)`

## Collegamento con gli script di training

Questo wrapper viene usato da:

- [train_sb3_sac.py](/Users/leonardo/Polito/FAIMDL/FAIML-RL-26_g44/part2/train_sb3_sac.py)
- [train_sb3_ppo.py](/Users/leonardo/Polito/FAIMDL/FAIML-RL-26_g44/part2/train_sb3_ppo.py)

La modalita' viene scelta dal flag:

```bash
--sampling-strategy {none,udr,adr}
```

## Limiti attuali

- `udr` randomizza massa e attriti, ma non randomizza ancora posizione target, rumore d'azione o rumore di osservazione
- `adr` adatta ancora solo il range di massa
- la logica ADR e' euristica e puo' essere raffinata se serve una versione piu' fedele alla consegna
