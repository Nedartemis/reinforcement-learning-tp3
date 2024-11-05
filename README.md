# Reinforcement Learning TP3

Indications dans le fichier taxi.py

## Authors

- sacha.hibon
- dorian.penso

## Implementation

Nous avons soigneusement suivi la méthode décrite dans cette vidéo :
https://www.youtube.com/watch?v=0iqz4tcKN58&list=PLMrJAkhIeNNQe1JXNvaFvURxGY4gE9k74&index=7

Ainsi, les apprentissages basés sur "qlearning" comporte de l'exploration tandis que les apprentissages basés sur "sarsa" n'en contient pas.

## Vidéos

Des vidéos prouvant le bon fonctionnement de nos modèles sont rangées dans le dossier "vidéos/".

## Efficacité

L'efficacité des différents modèles sont représentées dans les images rangées dans le dossier "plot/".

Le fichier "plot/curves_resume.png" donne une comparaison entre l'efficacité des différents modèles.
On remarque que :
- l'apprentissage par "qlearning" a les moins bonnes performances au final, c'est à dire les "total rewards" les moins élevées au final.
Ceci peut être expliqué par l'exploration qui est poursuivi même quand le modèle est bien entraîné.
- l'apprentissage par "sarsa" et par "qlearning_scheduling" atteignent les mêmes performances au final.
Ceci est tout à fait normal car l'exploration devient inexistante dans "qlearning_scheduling" lorsque le modèle a dépassé 10000 steps.
Ainsi, le comportement fini par être le même que "sarsa".
- l'apprentissage par "sarsa" est meilleur que "qlearning_scheduling" lui même meilleur que "qlearning".
Ainsi, l'exploration, dans cet environnement, ne consitue que du bruit et n'a pas de réel intérêt autre que faire converger le modèle plus lentement. 