# `part2/wrappers.py`

## Scopo

Questo file contiene i wrapper usati per modificare il comportamento dell'ambiente durante il training.

Al momento e' presente:

- `RewardShapingWrapper`

## Cosa fa `RewardShapingWrapper`

`RewardShapingWrapper` non cambia la fisica dell'ambiente e non modifica le transizioni.
Modifica solo il reward scalare restituito al policy learner.

La logica attuale e' questa:

1. parte dal reward originale dell'ambiente
2. sottrae una piccola penalita' per ogni step (`time_penalty`)
3. calcola la distanza tra oggetto e goal
4. se l'oggetto e' abbastanza vicino al goal, aggiunge un bonus fisso (`bonus`)

## Formula logica

Se indichiamo con:

- `r` il reward originale
- `p` la penalita' temporale
- `d` la distanza oggetto-goal
- `b` il bonus di vicinanza
- `t` la soglia di vicinanza (`bonus_distance`)

allora il reward finale e':

```text
r_shaped = r - p + bonus(d < t)
```

dove `bonus(d < t)` vale `b` solo quando `d < t`, altrimenti vale `0`.

## Il reward diventa piu' denso?

Sì, ma solo in modo limitato.

Il reward originale del task e' gia' `dense` nel file di training, quindi non e' un reward puramente sparso. Questo wrapper rende comunque il segnale un po' piu' informativo perche':

- introduce una penalita' ad ogni step, che spinge l'agente a risolvere il task piu' velocemente
- introduce un bonus quando l'oggetto entra vicino al goal

Quindi il reward non diventa "densissimo" in senso stretto, ma diventa piu' guidato rispetto al reward base.

Se volessi un reward davvero piu' denso, potresti aggiungere un termine proporzionale alla distanza, per esempio `-alpha * d`.

## Come funziona tecnicamente

La classe eredita da `gym.RewardWrapper`, quindi intercetta solo il reward.

### `_get_distance()`

Questa funzione prova a leggere:

- `self.env.unwrapped.task.get_achieved_goal()`
- `self.env.unwrapped.task.goal`

Se entrambe le informazioni esistono, calcola la distanza euclidea con `numpy.linalg.norm`.

### `reward()`

Questa funzione riceve il reward originale e restituisce il reward modificato.

La sequenza e':

- `shaped = reward - time_penalty`
- se la distanza e' disponibile e `dist < bonus_distance`, aggiunge `bonus`
- restituisce `shaped`

## Dove viene usato

Nel training PPO il wrapper viene applicato in `part2/train_sb3_ppo.py` prima di `VecNormalize`.

L'ordine e' importante:

1. ambiente base `PandaPush-v3`
2. `RandomizationWrapper`
3. `RewardShapingWrapper`
4. `VecNormalize` se attivato

## Limiti attuali

- il bonus e' discreto, non continuo
- non c'e' ancora un termine proporzionale alla distanza
- il wrapper dipende dall'API del task `Push` di `panda_gym`

## Possibile estensione

Se vuoi, si puo' rendere il reward davvero piu' denso aggiungendo un termine come:

```text
r_shaped = r - time_penalty - alpha * dist
```

oppure una versione piu' corretta dal punto di vista teorico con shaping potenziale:

```text
F(s, s') = gamma * Phi(s') - Phi(s)
```

