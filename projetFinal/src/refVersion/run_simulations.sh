#!/bin/bash

viscosities=("$@")

output_file="viscosity_drag_results.txt"


if [ -f "$output_file" ]; then
    rm "$output_file"
fi


for viscosity in "${viscosities[@]}"; do
    sed -i "s/^set viscosity.*$/set viscosity          = $viscosity/" parameters.prm
    
    output=$(./navier_stokes parameters.prm)
    
    drag_coefficient=$(echo "$output" | grep -oP 'Drag coefficient = \K[\d.]+')
    
    echo "$viscosity $drag_coefficient" >> "$output_file"
done

echo "Les simulations sont terminées. Les résultats sont sauvegardés dans '$output_file'."

