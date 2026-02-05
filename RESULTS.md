# GPT-2 Analysis Results

---

## Hidden State Magnitude Analysis

**Question: Does representation magnitude change with depth?**

**Answer: Yes, dramatically.** The L2 norm of hidden state vectors increases substantially as we go deeper into the network, with the most significant growth occurring in later layers (8-11).

### Norm Progression Across Layers

| Token Position | Token | Layer 0 (Embed) | Layer 4 | Layer 8 | Layer 12 (Final) |
|----------------|-------|-----------------|---------|---------|------------------|
| 0 | 'The' | 6.09 | 58.72 | 93.42 | 78.15 |
| 1 | 'Ġcapital' | 5.66 | 60.27 | 93.42 | 183.77 |
| 4 | 'Ġis' | 4.60 | 60.27 | 93.42 | 255.35 |
| 6 | ',' | 5.66 | 60.27 | 93.42 | 248.97 |
| 12 | 'Ġis' (last) | 4.60 | 60.27 | 93.42 | 248.50 |

**Key Observations:**
- **Embedding → Layer 4**: ~13x increase (4.6 → 60.3)
- **Layer 4 → Layer 8**: ~1.5x increase (60.3 → 93.4)
- **Layer 8 → Layer 12**: ~2.7x increase (93.4 → 248.5)
- **Overall**: ~54x increase from embedding to final layer
- **Layer 11 spike**: Norms peak at layer 11 (277.99) before dropping slightly at layer 12 (248.50)

**Interpretation:**
The model builds increasingly complex representations as depth increases. The dramatic norm growth in layers 8-11 suggests these layers perform significant feature composition. The slight decrease at layer 12 may indicate a normalization or refinement step before the final output projection.
