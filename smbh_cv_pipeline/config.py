from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Dict, List, Sequence

PARAM_NAMES = ["A", "R", "sigma", "B", "phi", "x0", "y0"]
ENV_PARAM_NAMES = ["j_A", "j_sx", "j_sy", "j_phi", "j_x0", "j_y0"]

DEFAULT_BOUNDS: Dict[str, List[float]] = {
    "A": [0.01, 0.07],
    "R": [50.0, 91.0],
    "sigma": [5.0, 10.0],
    "B": [0.0, 1.0],
    "phi": [-math.pi, math.pi],
    "x0": [128 - 30.0, 128 + 30.0],
    "y0": [128 - 30.0, 128 + 30.0],
    "j_A": [0.04, 0.10],
    "j_sx": [1.0, 8.0],
    "j_sy": [1.0, 8.0],
    "j_phi": [-math.pi / 2.0, math.pi / 2.0],
    "j_x0": [128 - 30.0, 128 + 30.0],
    "j_y0": [128 - 30.0, 128 + 30.0],
}

@dataclass
class PipelineConfig:
    out: str = "runs/smbh_cv_modular"
    seed: int = 239
    n_images: int = 100
    nx: int = 256
    ny: int = 256

    # Data generation
    dataset_mode: str = "simple_model"  # simple_model | degraded_model
    n_true_min: int = 3
    n_true_max: int = 10
    noise_sigma: float = 0.0
    beam_fwhm_px: float = 0.0
    noise_model: str = "gaussian"       # gaussian | correlated | none
    corr_noise_length_px: float = 2.0
    background_gradient: float = 0.0
    partial_ring_prob: float = 0.0
    overlap_mode: str = "random"        # random | clustered
    reconstruction_method: str = "none"

    true_bounds: Dict[str, List[float]] = field(default_factory=lambda: dict(DEFAULT_BOUNDS))
    fit_bounds: Dict[str, List[float]] = field(default_factory=lambda: dict(DEFAULT_BOUNDS))

    # D1
    d1_mask_method: str = "percentile_cc"      # none | percentile_cc | rect_max
    d1_edge_method: str = "legacy_laplace_positive"            # legacy_laplace_positive | log_mad | dog | sobel | laplace_sign
    d1_threshold_method: str = "positive"           # positive | mad | percentile | sauvola
    d1_morphology_method: str = "legacy_control_open"          # none | open_close | remove_small | legacy_control_open
    d1_edge_sigmas: str = "1.0,1.5,2.0,3.0"
    d1_tau: float = 3.0
    d1_percentile: float = 96.0
    d1_min_area: int = 4
    d1_skeletonize: bool = False
    d1_jet_percentile: float = 99.7
    d1_rect_half_size: int = 6
    d1_legacy_badpix: float = 0.07
    d1_legacy_max_iter: int = 20

    # D2
    d2_method: str = "sparse_rht_stratified"   # sparse_rht | sparse_rht_stratified
    d2_vote_weight: str = "mean"               # none | mean | product
    d2_votes_fraction: float = 0.25             # fraction of N^3, capped by d2_max_votes
    d2_max_votes: int = 300000
    d2_min_triangle_area: float = 15.0
    d2_center_bin_px: float = 1.0
    d2_radius_bin_px: float = 1.0
    d2_r_min: float = 30.0
    d2_r_max: float = 115.0
    d2_n_sectors: int = 12

    # D3
    d3_method: str = "ph_union_find_sparse"                # peak_nms | bootstrap_nms | ph_safe_proxy | ph_union_find_sparse | ph_cripser_dense
    d3_threshold_method: str = "bootstrap"      # relative | quantile | bootstrap
    d3_relative_threshold: float = 0.18
    d3_quantile: float = 0.985
    d3_bootstrap_quantile: float = 0.0
    d3_bootstrap_repeats: int = 24
    d3_nms_dx: float = 5.0
    d3_nms_dr: float = 4.0
    d3_max_candidates: int = 15
    d3_expected_max_rings: int = 12
    d3_top_k_for_ph: int = 5000
    d3_ph_memory_limit_gb: float = 8.0

    # D4
    d4_method: str = "profile_harmonic"        # profile_harmonic | constants
    d4_strip_half_width: float = 4.0
    d4_radial_window: float = 10.0
    d4_n_angle_bins: int = 96
    d4_max_init_candidates: int = 15

    # D5
    d5_engine: str = "scipy"                    # scipy
    d5_loss: str = "soft_l1"                    # linear | soft_l1 | huber | cauchy | arctan
    d5_residual_weighting: str = "ring_support" # uniform | ring_support | inverse_background
    d5_max_nfev: int = 1000
    d5_fit_background: bool = True
    d5_f_scale: float = 1.0
    d5_fit_stride: int = 8
    d5_local_then_global: bool = False

    # D6
    d6_method: str = "amplitude_merge_delta_bic"          # amplitude | delta_bic | merge_delta_bic | fdr_bh_amplitude | merge_fdr_bh_amplitude | amplitude_merge_delta_bic | amplitude_merge_fdr_bh_amplitude | none
    d6_min_artifacts: int = 0
    d6_amp_min: float = 0.0125
    d6_delta_bic_threshold: float = 7000.0
    d6_fdr_alpha: float = 0.05
    d6_fdr_no_covar_action: str = "keep_all"    # keep_all | amplitude | drop_min_amplitude
    d6_merge_center_px: float = 6.0
    d6_merge_radius_px: float = 6.0

    # Metrics
    max_match_cost: float = 1.25
    max_center_distance_px: float = 8.0
    max_center_error_fraction: float = 0.15
    max_radius_error_fraction: float = 0.15
    min_annular_iou: float = 0.10
    make_plots: bool = False
    save_arrays: bool = True

    # Visualization
    viz_dpi: int = 140
    viz_max_accumulator_points: int = 5000

    def to_dict(self):
        return asdict(self)


def parse_csv_floats(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]
