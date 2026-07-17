#!/usr/bin/env python3
"""
Test the spectral backend pipeline end-to-end.
"""

import sys
import numpy as np
from icm_tokenizer import ICMTokenizer
from matrix_builder import MatrixBuilder
from spectral import SpectralAnalyzer
from mode_router import ModeRouter


def test_pipeline():
    """Run end-to-end test."""
    test_asks = [
        "I want to build an AI that understands coherence",
        "Build AI want I coherence understand that",  # Incoherent
        "I want to build an AI system that measures reasoning coherence in real-time for research teams",  # Coherent
    ]

    tokenizer = ICMTokenizer()
    matrix_builder = MatrixBuilder()
    spectral_analyzer = SpectralAnalyzer()
    mode_router = ModeRouter()

    for ask in test_asks:
        print(f"\n{'='*60}")
        print(f"Ask: {ask}")
        print("=" * 60)

        try:
            # Step 1: Tokenize
            tokens = tokenizer.tokenize(ask)
            print(f"\nTokens ({len(tokens)}):")
            for i, token in enumerate(tokens[:10]):  # First 10
                print(f"  {i}: {token.text:20} | {token.type:10} | group={token.group_id}")
            if len(tokens) > 10:
                print(f"  ... and {len(tokens) - 10} more")

            # Step 2: Build matrix
            matrix, node_names = matrix_builder.build(tokens)
            print(f"\nMatrix: {matrix.shape[0]}x{matrix.shape[1]}")
            print(f"Non-zero entries: {np.count_nonzero(matrix)}")
            print(f"Matrix density: {np.count_nonzero(matrix) / (matrix.shape[0] * matrix.shape[1]):.2%}")

            # Step 3: Spectral analysis
            result = spectral_analyzer.analyze(matrix)
            print(f"\nSpectral Analysis:")
            print(f"  λ₁ (dominant eigenvalue): {result['lambda_1']:.4f}")
            print(f"  λ₂ (second eigenvalue):   {result['lambda_2']:.4f}")
            print(f"  Spectral gap:             {result['spectral_gap']:.4f}")
            print(f"  Coherence score:          {result['coherence']:.1f}/100")

            # Step 4: Mode activation
            mode_result = mode_router.activate(tokens, result["eigenvector_1"])
            print(f"\nMode Activation:")
            print(f"  Dominant mode: {mode_result['dominant_mode']}")
            print(f"  Mode entropy:  {mode_result['mode_entropy']:.4f}")
            print(f"  Top 3 modes:")
            mode_names = ModeRouter.mode_names()
            activations = list(enumerate(mode_result["mode_activation"]))
            activations.sort(key=lambda x: x[1], reverse=True)
            for mode_idx, activation in activations[:3]:
                print(f"    {mode_names[mode_idx]:12} {activation:.3f}")

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print("\n" + "=" * 60)
    print("All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(test_pipeline())
