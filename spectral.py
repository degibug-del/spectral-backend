"""
Spectral analysis: eigenvalue decomposition and coherence scoring.
"""

import numpy as np
from scipy.sparse.linalg import eigs
from scipy import sparse


class SpectralAnalyzer:
    """Compute eigenvalues, eigenvectors, and coherence from adjacency matrix."""

    # Calibration parameters (TBD after testing on 100-ask dataset)
    LAMBDA_MIN = 0.5      # Eigenvalue of incoherent ask
    LAMBDA_MAX = 8.0      # Eigenvalue of perfectly coherent ask
    CALIBRATED = False    # Set to True after calibration

    def analyze(
        self, matrix: np.ndarray
    ) -> dict:
        """
        Analyze adjacency matrix and return spectral properties.

        Returns:
            {
                "lambda_1": float,
                "lambda_2": float,
                "eigenvector_1": np.ndarray,
                "all_eigenvalues": np.ndarray,
                "coherence": float (0-100),
                "spectral_gap": float,
            }
        """
        # Ensure matrix is symmetric
        matrix = (matrix + matrix.T) / 2

        # Handle small matrices
        if matrix.shape[0] < 3:
            return self._handle_small_matrix(matrix)

        try:
            # Use sparse eigendecomposition for efficiency
            k = min(3, matrix.shape[0] - 1)  # Number of eigenvalues to compute
            eigenvalues, eigenvectors = eigs(
                sparse.csr_matrix(matrix), k=k, which="LM", return_eigenvectors=True
            )
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)

        except Exception:
            # Fall back to dense eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)
            # Sort by descending magnitude
            idx = np.argsort(-np.abs(eigenvalues))
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        # Dominant eigenvalue and eigenvector
        lambda_1 = float(np.max(np.abs(eigenvalues)))
        lambda_2 = float(np.abs(np.sort(np.abs(eigenvalues))[-2])) if len(eigenvalues) > 1 else 0
        idx_1 = np.argmax(np.abs(eigenvalues))
        v_1 = eigenvectors[:, idx_1]

        # Normalize eigenvector
        v_1 = v_1 / np.linalg.norm(v_1)

        # Coherence score
        coherence = self._coherence_score(lambda_1)

        # Spectral gap (λ₁ - λ₂)
        spectral_gap = lambda_1 - lambda_2 if lambda_2 > 0 else lambda_1

        return {
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "eigenvector_1": v_1,
            "all_eigenvalues": eigenvalues,
            "coherence": coherence,
            "spectral_gap": spectral_gap,
            "matrix_size": matrix.shape[0],
        }

    def _handle_small_matrix(self, matrix: np.ndarray) -> dict:
        """Handle matrices with < 3 nodes."""
        if matrix.shape[0] == 0:
            return {
                "lambda_1": 0,
                "lambda_2": 0,
                "eigenvector_1": np.array([]),
                "all_eigenvalues": np.array([]),
                "coherence": 0,
                "spectral_gap": 0,
                "matrix_size": 0,
            }

        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        lambda_1 = float(np.max(np.abs(eigenvalues)))
        lambda_2 = float(np.abs(np.sort(np.abs(eigenvalues))[-2])) if len(eigenvalues) > 1 else 0
        idx_1 = np.argmax(np.abs(eigenvalues))
        v_1 = eigenvectors[:, idx_1]
        v_1 = v_1 / np.linalg.norm(v_1)

        coherence = self._coherence_score(lambda_1)
        spectral_gap = lambda_1 - lambda_2 if lambda_2 > 0 else lambda_1

        return {
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "eigenvector_1": v_1,
            "all_eigenvalues": eigenvalues,
            "coherence": coherence,
            "spectral_gap": spectral_gap,
            "matrix_size": matrix.shape[0],
        }

    def _coherence_score(self, lambda_1: float) -> float:
        """
        Convert dominant eigenvalue to coherence score (0-100).

        Formula: coherence = 100 * (λ₁ - λ_min) / (λ_max - λ_min)

        Note: Calibration parameters (LAMBDA_MIN, LAMBDA_MAX) should be
        tuned against 100-ask hand-rated dataset.
        """
        if self.LAMBDA_MIN == self.LAMBDA_MAX:
            return 50.0

        score = 100 * (lambda_1 - self.LAMBDA_MIN) / (self.LAMBDA_MAX - self.LAMBDA_MIN)
        return float(np.clip(score, 0, 100))

    @classmethod
    def calibrate(cls, test_results: list[tuple[float, int]]) -> dict:
        """
        Calibrate coherence mapping from test data.

        Args:
            test_results: List of (lambda_1, human_rating) tuples

        Returns:
            {"lambda_min": float, "lambda_max": float, "correlation": float}
        """
        lambdas = np.array([x[0] for x in test_results])
        ratings = np.array([x[1] for x in test_results])

        # Linear calibration: find λ values at rating 0 and 100
        # For now, use percentile-based approach
        cls.LAMBDA_MIN = np.percentile(lambdas, 10)
        cls.LAMBDA_MAX = np.percentile(lambdas, 90)
        cls.CALIBRATED = True

        # Compute correlation
        predicted = [cls()._coherence_score(l) for l in lambdas]
        correlation = np.corrcoef(predicted, ratings)[0, 1]

        return {
            "lambda_min": float(cls.LAMBDA_MIN),
            "lambda_max": float(cls.LAMBDA_MAX),
            "correlation": float(correlation),
        }
