import numpy as np
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

# erreur de simulation E
S = 1.85
D = 1.25
E = S - D
print(f"L'erreur de simulation E est de {E:.2f}")

# u_D
sr = D * np.array([0.01, 0.05, 0.1, 0.2])
i_print = 2 # Index qui sera mis dans les print


br = 0.01 * 2 ** 0.5 
u_d = (sr ** 2 + br ** 2) ** 0.5
print("u_D = ", u_d)

u_val_left = (u_num ** 2 + u_input_left ** 2 + u_d ** 2) ** 0.5
u_val_right = (u_num ** 2 + u_input_right ** 2 + u_d ** 2) ** 0.5

print(f"L'erreur u_val_left est de {u_val_left[i_print]:.5f}")
print(f"L'erreur u_val_right est de {u_val_right[i_print]:.5f}")



# constante k avec 1-alpha = 0,954
k = 2

# Inequation erreur modele
# 𝐸 − ( 𝑘 𝑢_𝑣𝑎𝑙 ) ≤ 𝛿_𝑚𝑜𝑑𝑒𝑙 ≤ 𝐸 + ( 𝑘 𝑢_𝑣𝑎𝑙 )
delta_mod_inf = E - k * u_val_left
delta_mod_sup = E + k * u_val_right

delta_mod_milieu = (delta_mod_inf + delta_mod_sup) * 0.5
delta_mod_erreur = (delta_mod_inf - delta_mod_sup) * 0.5

print(f"𝛿_𝑚𝑜𝑑𝑒𝑙 est dans l'intervale : [{delta_mod_inf[i_print]:.4f} ; {delta_mod_sup[i_print]:.4f}]")
print(f"Autrement dit, 𝛿_𝑚𝑜𝑑𝑒𝑙 est ({delta_mod_milieu[i_print]:.4f} ± {-delta_mod_erreur[i_print]:.4f})")

E = 0.6001
C = 7
Uval = 0.2575

print("###Conclusion sur erreur modèle")

if E > C * Uval:
    print("delta_model = E")
elif C * Uval >= E >= Uval:
    print("|delta_model| < |E| + Uval \nEt signe(delta_model) = signe (E)")
elif Uval >= E >= Uval / C:
    print("|delta_model| < |E| + Uval \nEt pas de conclusion sur le signe de delta_model")
elif Uval/C >= E:
    print("|delta_model| < Uval \nEt pas de conclusion sur le signe de delta model")

# Create the plot
plt.figure(figsize=(6, 4))

colors = ['blue', 'green', 'red', 'purple']
labels = ['s_r = 1%', 's_r = 5%', 's_r = 10%', 's_r = 20%']

for i in range(0,4):
    print(i)
    plt.errorbar(x=i, y=E, yerr=[[E - delta_mod_inf[i]], [delta_mod_sup[i] - E]], fmt='o', 
             color=colors[i], capsize=5, capthick=2, elinewidth=2, label=labels[i])

# Setting y-axis limits
plt.ylim(-0.2, 1.2)

# Remove x-axis
plt.gca().axes.get_xaxis().set_visible(False)

# Setting labels and title
plt.ylabel('delta_model')
plt.title('Visualisation de l\'erreur du modèle')
plt.legend()

# Add a horizontal line at y = 0, make it black and thicker than the grid lines
plt.axhline(y=0, color='black', linewidth=2)

# Add a grid for better readability of the y axis
plt.grid(True)

# Save the figure to a file
plt.savefig('/path/to/Erreur_model_plot.png')
plt.close()  # Close the plot to free up memory

# This saves the plot as 'Erreur_model_plot.png' in the specified directory
