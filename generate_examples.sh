#!/bin/bash

# Script to generate CryptoPunks examples with concept vectors
# Usage: bash generate_examples.sh

CHECKPOINT="models/CRYPTOPUNKS/cryptopunks1/ckpt_final.pt"
OUTPUT_DIR="output/examples"

mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "🎨 Generating CryptoPunks Examples with Concepts"
echo "=================================================="
echo ""

# 1. Baseline - No accessories
echo "📌 [1/10] Generating baseline (no accessories)..."
python src/generate_with_concepts.py \
  --checkpoint "$CHECKPOINT" \
  --n_samples 16 \
  --seed 42 \
  --output "$OUTPUT_DIR/00_baseline.png"

# 2. Single accessories
echo "📌 [2/10] Generating with Cap..."
python src/generate_with_concepts.py \
  --checkpoint "$CHECKPOINT" \
  --accessories cap \
  --n_samples 16 \
  --seed 42 \
  --output "$OUTPUT_DIR/01_cap.png"

echo "📌 [3/10] Generating with Pipe..."
python src/generate_with_concepts.py \
  --checkpoint "$CHECKPOINT" \
  --accessories pipe \
  --n_samples 16 \
  --seed 42 \
  --output "$OUTPUT_DIR/02_pipe.png"

echo "📌 [4/10] Generating with Cigarette..."
python src/generate_with_concepts.py \
  --checkpoint "$CHECKPOINT" \
  --accessories cigarette \
  --n_samples 16 \
  --seed 42 \
  --output "$OUTPUT_DIR/03_cigarette.png"

echo "📌 [5/10] Generating with Hoodie..."
python src/generate_with_concepts.py \
  --checkpoint "$CHECKPOINT" \
  --accessories hoodie \
  --n_samples 16 \
  --seed 42 \
  --output "$OUTPUT_DIR/04_hoodie.png"

echo "📌 [6/10] Generating with Shades..."
python src/generate_with_concepts.py \
  --checkpoint "$CHECKPOINT" \
  --accessories shades \
  --n_samples 16 \
  --seed 42 \
  --output "$OUTPUT_DIR/05_shades.png"

# 3. Combined accessories
echo "📌 [7/10] Generating with Cap + Pipe..."
python src/generate_with_concepts.py \
  --checkpoint "$CHECKPOINT" \
  --accessories cap pipe \
  --n_samples 16 \
  --seed 42 \
  --output "$OUTPUT_DIR/06_cap_pipe.png"

echo "📌 [8/10] Generating with Shades + Cigarette..."
python src/generate_with_concepts.py \
  --checkpoint "$CHECKPOINT" \
  --accessories shades cigarette \
  --n_samples 16 \
  --seed 42 \
  --output "$OUTPUT_DIR/07_shades_cigarette.png"

echo "📌 [9/10] Generating with Hoodie + Cap..."
python src/generate_with_concepts.py \
  --checkpoint "$CHECKPOINT" \
  --accessories hoodie cap \
  --n_samples 16 \
  --seed 42 \
  --output "$OUTPUT_DIR/08_hoodie_cap.png"

# 4. Intensity variations
echo "📌 [10/10] Generating Cap with different intensities..."
python src/generate_with_concepts.py \
  --checkpoint "$CHECKPOINT" \
  --accessories cap \
  --concept_scale 0.5 \
  --n_samples 16 \
  --seed 42 \
  --output "$OUTPUT_DIR/09_cap_subtle.png"

python src/generate_with_concepts.py \
  --checkpoint "$CHECKPOINT" \
  --accessories cap \
  --concept_scale 2.0 \
  --n_samples 16 \
  --seed 42 \
  --output "$OUTPUT_DIR/10_cap_strong.png"

echo ""
echo "=================================================="
echo "✅ Generation complete!"
echo "📁 Examples saved to: $OUTPUT_DIR/"
echo "=================================================="
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.png
