# Calcul des erreurs
# Derniere modification : 03/31/2024
# Auteurs : - Pablo ELICES PAZ
#           - Lucas BRAHIC
#           - Justin BELZILE

# u_num
u_num = 1

# u_input
u_input = 1

# u_D
u_d = 1

u_val = (u_num ** 2 + u_input ** 2 + u_d ** 2) ** 0.5

print(f"L'erreur u_val est de {u_val:.2f}")

# erreur de simulation E
S = 1
D = 1
E = S - D
print(f"L'erreur de simulation E est de {E:.2f}")

# constante k avec 1-alpha = 0,954
k = 2

# Inequation erreur modele
# 𝐸 − ( 𝑘 𝑢_𝑣𝑎𝑙 ) ≤ 𝛿_𝑚𝑜𝑑𝑒𝑙 ≤ 𝐸 + ( 𝑘 𝑢_𝑣𝑎𝑙 )
delta_mod_inf = E - k * u_val
delta_mod_sup = E + k * u_val

delta_mod_milieu = (delta_mod_inf + delta_mod_sup) * 0.5
delta_mod_erreur = (delta_mod_inf - delta_mod_sup) * 0.5

print(f"𝛿_𝑚𝑜𝑑𝑒𝑙 est dans l'intervale : [{delta_mod_inf:.2f} ; {delta_mod_sup:.2f}]")
print(f"Autrement dit, 𝛿_𝑚𝑜𝑑𝑒𝑙 est ({delta_mod_milieu:.2f} ± {-delta_mod_erreur:.2f})")
