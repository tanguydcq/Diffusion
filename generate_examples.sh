#!/bin/bash
# Script de génération d'exemples de CryptoPunks avec différents types et accessoires
# Basé sur le fichier metadata.json

echo "🎨 Génération d'exemples de CryptoPunks avec concept steering"
echo "=============================================================="
echo ""

# Créer le répertoire de sortie
mkdir -p output/examples

# Utiliser Python du venv
PYTHON=".venv/bin/python"

# Types disponibles: Alien (0), Ape (1), Female (2), Male (3), Zombie (4)

# ===== ALIENS =====
echo "👽 Génération d'Aliens..."

# Alien avec pipe et lunettes
$PYTHON generate_by_phrase.py Alien Pipe "Classic Shades" \
    --strength 1.8 --n 8 --output output/examples/alien_pipe_shades.png

# Alien avec bonnet et collier
$PYTHON generate_by_phrase.py Alien "Knitted Cap" "Gold Chain" \
    --strength 1.5 --n 8 --output output/examples/alien_cap_chain.png

# ===== APES =====
echo "🦍 Génération d'Apes..."

# Ape avec mohawk et cigarette
$PYTHON generate_by_phrase.py Ape Mohawk Cigarette \
    --strength 1.7 --n 8 --output output/examples/ape_mohawk_cig.png

# Ape avec casquette et boucle d'oreille
$PYTHON generate_by_phrase.py Ape "Cap Forward" Earring \
    --strength 1.6 --n 8 --output output/examples/ape_cap_earring.png

# ===== FEMALES =====
echo "👩 Génération de Females..."

# Female avec tiare et rouge à lèvres
$PYTHON generate_by_phrase.py Female Tiara "Hot Lipstick" \
    --strength 1.5 --n 8 --output output/examples/female_tiara_lipstick.png

# Female avec bob blond et collier
$PYTHON generate_by_phrase.py Female "Blonde Bob" Choker \
    --strength 1.6 --n 8 --output output/examples/female_bob_choker.png

# Female avec cheveux violets et ombre à paupières
$PYTHON generate_by_phrase.py Female "Purple Hair" "Purple Eye Shadow" \
    --strength 1.7 --n 8 --output output/examples/female_purple_style.png

# Female avec lunettes nerd et rouge à lèvres noir
$PYTHON generate_by_phrase.py Female "Nerd Glasses" "Black Lipstick" \
    --strength 1.8 --n 8 --output output/examples/female_nerd_goth.png

# ===== MALES =====
echo "👨 Génération de Males..."

# Male avec grande barbe et lunettes
$PYTHON generate_by_phrase.py Male "Big Beard" "Big Shades" \
    --strength 1.6 --n 8 --output output/examples/male_beard_shades.png

# Male avec chapeau cowboy et pipe
$PYTHON generate_by_phrase.py Male "Cowboy Hat" Pipe \
    --strength 1.7 --n 8 --output output/examples/male_cowboy_pipe.png

# Male avec casque de police et moustache
$PYTHON generate_by_phrase.py Male "Police Cap" Mustache \
    --strength 1.5 --n 8 --output output/examples/male_cop_mustache.png

# Male avec haut-de-forme et barbe luxueuse
$PYTHON generate_by_phrase.py Male "Top Hat" "Luxurious Beard" \
    --strength 1.8 --n 8 --output output/examples/male_tophat_luxbeard.png

# ===== ZOMBIES =====
echo "🧟 Génération de Zombies..."

# Zombie avec masque médical et bandana
$PYTHON generate_by_phrase.py Zombie "Medical Mask" Bandana \
    --strength 1.6 --n 8 --output output/examples/zombie_mask_bandana.png

# Zombie avec casque pilote et lunettes de soudage
$PYTHON generate_by_phrase.py Zombie "Pilot Helmet" "Welding Goggles" \
    --strength 1.9 --n 8 --output output/examples/zombie_pilot_goggles.png

# ===== COMBINAISONS CRÉATIVES =====
echo "🎭 Génération de combinaisons créatives..."

# Alien punk avec mohawk rouge et VR
$PYTHON generate_by_phrase.py Alien "Red Mohawk" VR "Gold Chain" \
    --strength 2.0 --n 8 --output output/examples/alien_cyberpunk.png

# Female pirate avec bandana et eye patch
$PYTHON generate_by_phrase.py Female Bandana "Eye Patch" Smile \
    --strength 1.7 --n 8 --output output/examples/female_pirate.png

# Male hipster avec beanie, barbe normale et vape
$PYTHON generate_by_phrase.py Male Beanie "Normal Beard" Vape \
    --strength 1.6 --n 8 --output output/examples/male_hipster.png

# Zombie clown avec nez de clown et cheveux verts
$PYTHON generate_by_phrase.py Zombie "Clown Nose" "Clown Hair Green" \
    --strength 2.0 --n 8 --output output/examples/zombie_clown.png

# Ape gangster avec do-rag, chaîne d'or et cigarette
$PYTHON generate_by_phrase.py Ape "Do-rag" "Gold Chain" Cigarette \
    --strength 1.8 --n 8 --output output/examples/ape_gangster.png

# ===== TESTS PAR INDEX =====
echo "🔢 Génération par index d'accessoires..."

# Utilisation d'index directs (plus rapide)
$PYTHON generate_by_phrase.py Male 45 14 33 \
    --strength 1.7 --n 8 --output output/examples/male_by_index.png
# Index 45=Mohawk, 14=Cigarette, 33=Gold Chain

$PYTHON generate_by_phrase.py Female 78 40 13 \
    --strength 1.6 --n 8 --output output/examples/female_by_index.png
# Index 78=Tiara, 40=Hot Lipstick, 13=Choker

# ===== STYLES EXTRÊMES =====
echo "⚡ Génération avec force extrême..."

# Force très élevée pour un style marqué
$PYTHON generate_by_phrase.py Zombie "Crazy Hair" "Clown Eyes Green" Frown \
    --strength 2.5 --n 8 --output output/examples/zombie_extreme.png

# Force modérée pour subtilité
$PYTHON generate_by_phrase.py Female "Rosy Cheeks" Smile Mole \
    --strength 1.2 --n 8 --output output/examples/female_subtle.png

echo ""
echo "✅ Génération terminée ! Consultez le dossier output/examples/"
echo "📊 Total: ~25 variations générées"
