import matplotlib.pyplot as plt

# Calcul des erreurs
# Derniere modification : 03/31/2024
# Auteurs : - Pablo ELICES PAZ
#           - Lucas BRAHIC
#           - Justin BELZILE

# u_num
u_num = 0.00024

# u_input
u_input_left = 0.02719
u_input_right = 0.02759

# u_D
u_d = 0.02

u_val_left = (u_num ** 2 + u_input_left ** 2 + u_d ** 2) ** 0.5
u_val_right = (u_num ** 2 + u_input_right ** 2 + u_d ** 2) ** 0.5

print(f"L'erreur u_val_left est de {u_val_left:.5f}")
print(f"L'erreur u_val_right est de {u_val_right:.5f}")

# erreur de simulation E
S = 1.85
D = 1.25
E = S - D
print(f"L'erreur de simulation E est de {E:.2f}")

# constante k avec 1-alpha = 0,954
k = 2

# Inequation erreur modele
# 𝐸 − ( 𝑘 𝑢_𝑣𝑎𝑙 ) ≤ 𝛿_𝑚𝑜𝑑𝑒𝑙 ≤ 𝐸 + ( 𝑘 𝑢_𝑣𝑎𝑙 )
delta_mod_inf = E - k * u_val_left
delta_mod_sup = E + k * u_val_right

delta_mod_milieu = (delta_mod_inf + delta_mod_sup) * 0.5
delta_mod_erreur = (delta_mod_inf - delta_mod_sup) * 0.5

print(f"𝛿_𝑚𝑜𝑑𝑒𝑙 est dans l'intervale : [{delta_mod_inf:.4f} ; {delta_mod_sup:.4f}]")
print(f"Autrement dit, 𝛿_𝑚𝑜𝑑𝑒𝑙 est ({delta_mod_milieu:.4f} ± {-delta_mod_erreur:.4f})")

E = 0.6003
C = 10
Uval = 0.0678

print("###\nConclusion sur delta_model :\n")

if E > C * Uval:
    print("delta_model = E")
elif C * Uval >= E >= Uval:
    print("|delta_model| < |E| + Uval \nEt signe(delta_model) = signe (E)")
elif Uval >= E >= Uval / C:
    print("|delta_model| < |E| + Uval \nEt pas de conclusion sur le signe de delta_model")
elif Uval/C >= E:
    print("|delta_model| < Uval \nEt pas de conclusion sur le signe de delta model")


# Define the mean and confidence interval values
mean = E
conf_interval = (delta_mod_inf, delta_mod_sup)

# Create the plot
plt.figure(figsize=(6, 4))
plt.errorbar(x=0, y=mean, yerr=[[mean - conf_interval[0]], [conf_interval[1] - mean]], fmt='o', 
             color='blue', capsize=5, capthick=2, elinewidth=2)

# Setting y-axis limits
plt.ylim(-0.2, 0.8)

# Remove x-axis
plt.gca().axes.get_xaxis().set_visible(False)

# Setting labels and title
plt.ylabel('delta_model')
plt.title('Visualisation de l\'erreur du modèle')

# Add a horizontal line at y = 0, make it black and thicker than the grid lines
plt.axhline(y=0, color='black', linewidth=2)

# Add a grid for better readability of the y axis
plt.grid(True)

# Save the figure to a file
plt.savefig('/path/to/Erreur_model_plot.png')
plt.close()  # Close the plot to free up memory

# This saves the plot as 'Erreur_model_plot.png' in the specified directory
