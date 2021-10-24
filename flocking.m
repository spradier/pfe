clear all

# Initialisation des différents points (drones)
px = -300 + 100*randn(1,50);
py = -300 + 100*randn(1,50);
pz = -300 + 100*randn(1,50);

# Initialisation des vecteurs vitesse initiaux
vx = 10 * randn(1, 50);
vy = 10 * randn(1, 50);
vz = 10 * randn(1, 50);

# On met les points et les vitesse dans une matrice
points = [px' py' pz'];
vit = [vx' vy' vz'];

# Calcul du centroid et de la vitesse moyenne initiale
centroid = [mean(px) mean(py) mean(pz)];
v_moy = [mean(vx) mean(vy) mean(vz)];

# Définition des coefficients
k1 = 0.5; k2 = 0.3; k3 = 0.3;

figure(1), clf

# Pour chaque unité de temps
for t=1:100
  # Pour chaque points
  for p=1:length(points)
    
    # Force d'attraction
    F1 = k1 * (centroid - points(p, :));
    # Force de cohésion des vitesses
    F2 = k2 * v_moy;
    
    # Liste des forces de répulsion entre 1 point et tous les autres
    rep = [];
    # Pour chaque point (appariement 2 par 2)
    for j=1:length(points)
      # On ne compare pas un point avec lui même
      if p!=j
        # Calcul de distance entre 2 points
        r = points(p, :) - points(j, :);
        # Norme du rayon
        r_norm = sqrt(r(1)^2 + r(2)^2 + r(3)^2);
        # Force (vecteur / norme au cube)
        F = (r / r_norm^3);
        # On ajoute dans la liste
        rep = [rep F'];
       end
    endfor
    
    # Force finale F3 par la moyenne des forces entre chaque point
    F3 = k3 * [mean(rep(1, :)) mean(rep(2, :)) mean(rep(3, :))];
    
    # New coord c'est le point déplacé de la somme des forces
    new_coords = points(p, :) + F1 + F2 + F3;
    points(p, 1) = new_coords(1); points(p, 2) = new_coords(2); points(p, 3) = new_coords(3);
  endfor
  plot3(points(:, 1), points(:, 2), points(:, 3), 'o'), grid on, zoom on
  drawnow
endfor

