# Calcul des erreurs
# Derniere modification : 03/24/2024
# Auteurs : - Pablo ELICES PAZ
#           - Lucas BRAHIC
#           - Justin BELZILE

# u_num
u_num = 0.1177

# u_input
u_input_gauche = 7.76 ** 0.5
u_input_droite = 11.21 ** 0.5

# u_D
sr = 14.7
br = 10.0
u_d = (sr ** 2 + br ** 2) ** 0.5

u_val_gauche = (u_num ** 2 + u_input_gauche ** 2 + u_d ** 2) ** 0.5
u_val_droite = (u_num ** 2 + u_input_droite ** 2 + u_d ** 2) ** 0.5

print(f"L'erreur u_val a gauche est de {u_val_gauche:.2f} micrometres^2")
print(f"L'erreur u_val a droite est de {u_val_droite:.2f} micrometres^2")


# erreur de simulation E
S = 25.1
D = 80.6
E = S - D
print(f"L'erreur de simulation E est de {E:.2f} micrometres^2")

# constante k avec 1-alpha = 0,954
k = 2

# Inequation erreur modele
# 𝐸 − ( 𝑘 𝑢_𝑣𝑎𝑙 ) ≤ 𝛿_𝑚𝑜𝑑𝑒𝑙 ≤ 𝐸 + ( 𝑘 𝑢_𝑣𝑎𝑙 )
delta_mod_inf = E - k * u_val_gauche
delta_mod_sup = E + k * u_val_droite

delta_mod_milieu = (delta_mod_inf + delta_mod_sup) * 0.5
delta_mod_erreur = (delta_mod_inf - delta_mod_sup) * 0.5

print(f"L'erreur du modele est dans l'intervale : [{delta_mod_inf:.2f} ; {delta_mod_sup:.2f}] micrometres^2")
print(f"Autrement dit, delta_model est ({delta_mod_milieu:.2f} ± {-delta_mod_erreur:.2f}) micrometres^2")
