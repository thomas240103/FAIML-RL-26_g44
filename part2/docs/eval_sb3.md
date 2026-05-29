# `part2/eval_sb3.py`

## Scopo

Questo script valuta un modello salvato su `PandaPush-v3` e stampa statistiche aggregate sulle performance.

## Cosa fa

Per ogni episodio:

1. resetta l'ambiente
2. lascia che il modello predica le azioni
3. accumula il reward totale
4. registra il successo finale se `info["is_success"]` e' presente

Alla fine stampa:

- numero episodi
- return medio
- deviazione standard
- return minimo
- return massimo
- success rate
- massa minima/media/massima usata in evaluation
- success rate sugli oggetti pesanti
- metriche aggregate per bin di massa, se la massa varia

## Argomenti supportati

- `--model-path`: path al file `.zip`
- `--algo`: algoritmo del modello (`sac`, `ppo`), opzionale se il filename lo rende inferibile
- `--episodes`: numero di episodi di evaluation
- `--stochastic`: usa la policy in modo non deterministico
- `--render`: apre il rendering umano
- `--env-type`: `source` oppure `target`
- `--max-episode-steps`: orizzonte esplicito dell'episodio di eval
- `--seed`: seed opzionale per rendere riproducibile la sequenza di episodi
- `--eval-mass-mode`: modalita' massa (`fixed`, `uniform`, `grid`)
- `--fixed-mass`: massa fissa custom, altrimenti usa la massa nominale del dominio
- `--mass-min`, `--mass-max`: range della massa per `uniform` e `grid`
- `--mass-values`: lista esplicita di masse per `grid`, ad esempio `5,6,7,8,9,10`
- `--mass-grid-size`: numero di masse equispaziate se `--mass-values` non e' passato
- `--mass-bins`: numero di bin per il riepilogo per massa
- `--heavy-mass-threshold`: soglia per il success rate sugli oggetti pesanti
- `--eval-friction-mode`: modalita' attrito (`fixed`, `uniform`)
- `--object-lateral-friction`: attrito laterale fisso dell'oggetto
- `--table-lateral-friction`: attrito laterale fisso del tavolo
- `--object-spinning-friction`: attrito rotazionale fisso dell'oggetto
- `--object-lateral-friction-min`, `--object-lateral-friction-max`: range dell'attrito laterale dell'oggetto
- `--table-lateral-friction-min`, `--table-lateral-friction-max`: range dell'attrito laterale del tavolo
- `--object-spinning-friction-min`, `--object-spinning-friction-max`: range dell'attrito rotazionale dell'oggetto
- `--friction-bins`: numero di bin per il riepilogo per attrito

## Massa in evaluation

Il default resta una evaluation a massa fissa:

- `--env-type source` usa massa nominale `1.0 kg`
- `--env-type target` usa massa nominale `5.0 kg`

Per rendere l'evaluation piu' difficile si puo' campionare una massa diversa a ogni episodio:

```bash
python part2/eval_sb3.py \
  --model-path models/SAC_none_source_500k/model.zip \
  --algo sac \
  --env-type target \
  --episodes 500 \
  --eval-mass-mode uniform \
  --mass-min 5 \
  --mass-max 10 \
  --seed 42
```

Per una stress evaluation deterministica, usare una griglia:

```bash
python part2/eval_sb3.py \
  --model-path models/SAC_none_source_500k/model.zip \
  --algo sac \
  --env-type target \
  --episodes 600 \
  --eval-mass-mode grid \
  --mass-values 5,6,7,8,9,10 \
  --seed 42
```

La griglia cicla sui valori indicati. Con `600` episodi e 6 masse, ogni massa viene valutata 100 volte.

## Attrito in evaluation

Per il task di pushing, l'attrito e' spesso piu' rilevante della sola massa: cambia quanto l'oggetto scivola, quanto ruota e quanta forza effettiva serve per muoverlo.

Il default resta `--eval-friction-mode fixed`, cioe' usa i valori nominali di PyBullet. Per randomizzare l'attrito a ogni episodio:

```bash
python part2/eval_sb3.py \
  --model-path models/SAC_none_source_500k/model.zip \
  --algo sac \
  --env-type target \
  --episodes 500 \
  --eval-mass-mode uniform \
  --mass-min 5 \
  --mass-max 10 \
  --eval-friction-mode uniform \
  --object-lateral-friction-min 0.2 \
  --object-lateral-friction-max 1.5 \
  --table-lateral-friction-min 0.2 \
  --table-lateral-friction-max 1.5 \
  --object-spinning-friction-min 0.0 \
  --object-spinning-friction-max 0.01 \
  --seed 42
```

Lo script stampa per ogni episodio massa, attrito laterale dell'oggetto, attrito laterale del tavolo e attrito rotazionale dell'oggetto. Nel riepilogo finale stampa anche statistiche aggregate per bin di attrito.

## Orizzonte

L'orizzonte e' ora esplicito con `--max-episode-steps` e il default e' `500`, coerente con gli script di training principali. Questo evita confronti ambigui con il default registrato di `PandaPush-v3`, che puo' essere piu' corto.

## Ruolo negli esperimenti

Questo script e' quello che usi per confronti come:

- `source -> source`
- `source -> target`
- `target -> target`

Per esempio:

- alleni `sac_push_none_source_500k.zip`
- lo valuti su `source` per la baseline
- lo valuti su `target` per il lower bound di transfer

## Esempi d'uso

```bash
python part2/eval_sb3.py --model-path sac_push_none_source_500k.zip --env-type source --episodes 100
python part2/eval_sb3.py --model-path sac_push_none_source_500k.zip --env-type target --episodes 100
python part2/eval_sb3.py --model-path sac_push_none_target_500k.zip --env-type target --episodes 100
```

## Nota sulla determinicita'

Per risultati confrontabili conviene usare il default deterministico.

L'opzione `--stochastic` e' utile soprattutto per debug o per capire quanto la policy dipenda dalla parte esplorativa.
