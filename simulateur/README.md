Pour AirSim voilà la démarche :

Etape 1 : Installer AirSim (soit sur ton pc, soit sur les pcs de l'école en voyant avec le service info)
lien : https://microsoft.github.io/AirSim/build_windows/

Etape 2 : Prise en main d'AirSim
lien : https://microsoft.github.io/AirSim/

Concernant la prise en main, si on fait par étape :
1) Générer une map (en prenant une map qui existe bien sûr)
2) Générer un drone sur la map à une coordonnées spécifiques (en prenant un drone qui existe)
3) Lui placer des capteurs (caméras basiques, ou quelque chose qui se rapproche de ce qu'on utilise disponible dans la librairie des capteurs du logiciel)

Tu as une liste de tutoriels sur le lien que je te donne (en descendant un peu), regarde déjà globalement comment sont organisés les codes (comment ils génèrent la map, comment ils instancient un drone, comment ils lui placent des capteurs) et tu reprends ça dans ton script.

Si tu arrives à faire ça déjà, et à la finir, c'est énorme parce qu'on pourra directement implémenté ce qu'on a fait dessus à savoir :
- le flocking
- la stereo
- les fisheyes

Bonne chance
